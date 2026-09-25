"""
Production-grade Submission Generator for Amazon ML Challenge 2026.
Generates output/matching_results.tsv and output/candidate_pairs.tsv
for all 1,732,544 test entities across US, India, and France.
"""

import os
import sys
import gc
import time
import subprocess
import collections
from typing import Dict, List, Set, Tuple, Optional
import pandas as pd
import numpy as np
from tqdm import tqdm

# Ensure package modules can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess import normalize_business_name, normalize_address
from src.blocking import MultiIndexBlocker
from src.features import extract_pair_features
from src.model import EntityMatcherModel, compute_entity_f05


def load_train_aligned_data(
    train_dir: str,
    n_train_s1: int = 15000,
    n_val_s1: int = 1500,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], Dict[str, Set[str]], List[Dict[str, str]]]:
    """
    Load S1 training and validation reference records, ground truth, and target positives.
    """
    total_needed = n_train_s1 + n_val_s1
    print(f"\n[1/5] Loading {total_needed:,} Source 1 training records...")
    s1_df = pd.read_csv(os.path.join(train_dir, "train_source1.tsv"), sep="\t", nrows=total_needed)
    for col in s1_df.columns:
        s1_df[col] = s1_df[col].fillna("").astype(str)
    s1_all = s1_df.to_dict(orient="records")
    s1_needed_ids = set(s1_df["entity_id"])

    print("Aligning ground truth for sampled S1 entities...")
    gt = {}
    needed_target_ids = set()
    for chunk in pd.read_csv(os.path.join(train_dir, "train_ground_truth.tsv"), sep="\t", chunksize=150000):
        sub = chunk[chunk["source1_entity_id"].isin(s1_needed_ids)]
        for _, r in sub.iterrows():
            s1 = r["source1_entity_id"]
            m_str = str(r["matched_entity_ids"]).strip() if pd.notna(r["matched_entity_ids"]) else ""
            if m_str and m_str.lower() != "nan":
                m_set = set(m.strip() for m in m_str.split(",") if m.strip())
                gt[s1] = m_set
                needed_target_ids.update(m_set)
            else:
                gt[s1] = set()
        if len(gt) >= len(s1_needed_ids):
            break

    print(f"Ground truth aligned for {len(gt):,} entities ({len(needed_target_ids):,} true match targets).")

    # Split into Train (15k) and Val (1.5k)
    np.random.seed(42)
    indices = np.random.permutation(len(s1_all))
    train_s1 = [s1_all[i] for i in indices[:n_train_s1]]
    val_s1 = [s1_all[i] for i in indices[n_train_s1:]]
    val_gt = {r["entity_id"]: gt.get(r["entity_id"], set()) for r in val_s1}

    # Stream S2 and S3 to collect all target positives + sample negatives
    print(f"Scanning target sources for {len(needed_target_ids):,} positive targets...")
    train_targets = []
    for src in ["train_source2.tsv", "train_source3.tsv"]:
        fpath = os.path.join(train_dir, src)
        count = 0
        neg_budget = 10000
        for chunk in pd.read_csv(fpath, sep="\t", chunksize=200000):
            for col in chunk.columns:
                chunk[col] = chunk[col].fillna("").astype(str)
            pos_chunk = chunk[chunk["entity_id"].isin(needed_target_ids)]
            if len(pos_chunk) > 0:
                train_targets.extend(pos_chunk.to_dict(orient="records"))
                count += len(pos_chunk)
            if neg_budget > 0:
                take = min(neg_budget, len(chunk))
                train_targets.extend(chunk.iloc[:take].to_dict(orient="records"))
                neg_budget -= take
        print(f"  Loaded {count:,} positive matches from {src}")

    print(f"Total training target pool: {len(train_targets):,} records.")
    return train_s1, val_s1, val_gt, gt, train_targets


def train_matching_model(
    train_s1: List[Dict[str, str]],
    val_s1: List[Dict[str, str]],
    val_gt: Dict[str, Set[str]],
    all_gt: Dict[str, Set[str]],
    train_targets: List[Dict[str, str]],
) -> Tuple[EntityMatcherModel, float]:
    """
    Build training and validation feature matrices, fit LightGBM, and calibrate threshold.
    """
    print("\n[2/5] Building features and training matching model...")
    # Group by country
    targets_by_ctry = collections.defaultdict(list)
    for r in train_targets:
        targets_by_ctry[r["country"]].append(r)

    train_s1_by_ctry = collections.defaultdict(list)
    for r in train_s1:
        train_s1_by_ctry[r["country"]].append(r)

    val_s1_by_ctry = collections.defaultdict(list)
    for r in val_s1:
        val_s1_by_ctry[r["country"]].append(r)

    blocker = MultiIndexBlocker(max_df_token=4000, max_candidates_per_entity=30)
    X_train_list, y_train_list = [], []
    X_val_list, val_pair_ids = [], []

    for ctry in ["US", "India"]:
        tgt = targets_by_ctry.get(ctry, [])
        tr = train_s1_by_ctry.get(ctry, [])
        vl = val_s1_by_ctry.get(ctry, [])

        if not tgt or not tr:
            continue

        print(f"  Country {ctry}: {len(tgt):,} targets, {len(tr):,} train S1, {len(vl):,} val S1")
        blocker.index_targets(tgt)

        # 1. Train candidates (inject true matches so model learns positive characteristics)
        tr_cands = blocker.retrieve_candidates(tr)
        for r in tr:
            eid = r["entity_id"]
            for tp in all_gt.get(eid, set()):
                if tp in blocker.target_records and tp not in tr_cands.get(eid, []):
                    tr_cands[eid].append(tp)

        tr_lookup = {r["entity_id"]: {"norm_name": normalize_business_name(r["business_name"]), "norm_addr": normalize_address(r["business_address"])} for r in tr}
        
        # Build features for train
        feats_tr, pairs_tr = [], []
        for s1_id, cands in tr_cands.items():
            s1_d = tr_lookup[s1_id]
            for cid in cands:
                cd = blocker.target_records[cid]
                feats_tr.append(extract_pair_features(s1_d["norm_name"], s1_d["norm_addr"], cd["norm_name"], cd["norm_addr"], cid))
                pairs_tr.append((s1_id, cid))

        y_tr = np.array([1 if p[1] in all_gt.get(p[0], set()) else 0 for p in pairs_tr], dtype=np.int32)
        X_train_list.append(np.array(feats_tr, dtype=np.float32))
        y_train_list.append(y_tr)

        # 2. Val candidates (Strict raw candidate generation, NO injection)
        vl_cands = blocker.retrieve_candidates(vl)
        vl_lookup = {r["entity_id"]: {"norm_name": normalize_business_name(r["business_name"]), "norm_addr": normalize_address(r["business_address"])} for r in vl}
        
        feats_vl, pairs_vl = [], []
        for s1_id, cands in vl_cands.items():
            s1_d = vl_lookup[s1_id]
            for cid in cands:
                cd = blocker.target_records[cid]
                feats_vl.append(extract_pair_features(s1_d["norm_name"], s1_d["norm_addr"], cd["norm_name"], cd["norm_addr"], cid))
                pairs_vl.append((s1_id, cid))

        X_val_list.append(np.array(feats_vl, dtype=np.float32))
        val_pair_ids.extend(pairs_vl)

    X_train = np.vstack(X_train_list)
    y_train = np.concatenate(y_train_list)
    X_val = np.vstack(X_val_list)

    print(f"Total training pairs: {len(X_train):,} (Positives: {np.sum(y_train):,} / {len(y_train):,} = {np.mean(y_train)*100:.2f}%)")
    print(f"Total validation pairs: {len(X_val):,}")

    model = EntityMatcherModel(threshold=0.75)
    model.fit(X_train, y_train)

    best_thresh = model.optimize_threshold(
        X_val, val_pair_ids, val_gt,
        threshold_candidates=[0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
    )

    # Detailed validation metrics
    probs = model.predict_proba(X_val)
    val_pred_matches = collections.defaultdict(set)
    for p, (s1, c) in zip(probs, val_pair_ids):
        if p >= best_thresh:
            val_pred_matches[s1].add(c)

    f05_scores = [compute_entity_f05(val_gt[s1], val_pred_matches.get(s1, set())) for s1 in val_gt]
    print("\n" + "=" * 60)
    print(f"VALIDATION PERFORMANCE SUMMARY")
    print(f"  Macro F_0.5 Score: {np.mean(f05_scores):.4f}")
    print(f"  Decision Threshold: {best_thresh:.2f}")
    print("=" * 60)

    # Free memory
    del X_train, y_train, X_val, blocker
    gc.collect()

    return model, best_thresh


def run_country_test_inference(
    country: str,
    test_dir: str,
    model: EntityMatcherModel,
    threshold: float,
    candidate_file_handle,
    matching_file_handle,
):
    """
    Run full blocking and inference for one country partition and stream results to disk.
    """
    print(f"\n[3/5] Starting Inference for Country: {country}")
    t0 = time.time()

    # 1. Load test S1 records for this country
    print(f"  Loading test S1 records for {country}...")
    s1_records = []
    chunk_size = 200000
    for chunk in pd.read_csv(os.path.join(test_dir, "test_source1.tsv"), sep="\t", chunksize=chunk_size):
        sub = chunk[chunk["country"] == country]
        if len(sub) > 0:
            for col in sub.columns:
                sub[col] = sub[col].fillna("").astype(str)
            s1_records.extend(sub.to_dict(orient="records"))

    print(f"  Total {country} S1 records: {len(s1_records):,}")

    # 2. Load test S2 and S3 target records for this country
    print(f"  Loading test S2 & S3 target records for {country}...")
    target_records = []
    for src in ["test_source2.tsv", "test_source3.tsv"]:
        fpath = os.path.join(test_dir, src)
        count = 0
        for chunk in pd.read_csv(fpath, sep="\t", chunksize=chunk_size):
            sub = chunk[chunk["country"] == country]
            if len(sub) > 0:
                for col in sub.columns:
                    sub[col] = sub[col].fillna("").astype(str)
                target_records.extend(sub.to_dict(orient="records"))
                count += len(sub)
        print(f"    Loaded {count:,} targets from {src}")

    print(f"  Total {country} targets: {len(target_records):,}")

    # 3. Build inverted indices
    blocker = MultiIndexBlocker(max_df_token=4000, max_candidates_per_entity=25)
    print(f"  Indexing {len(target_records):,} targets...")
    blocker.index_targets(target_records, show_progress=False)
    del target_records
    gc.collect()

    # 4. Query S1 entities in streaming batches
    batch_size = 50000
    total_s1 = len(s1_records)
    print(f"  Running candidate retrieval and inference over {total_s1:,} entities in batches of {batch_size:,}...")

    total_matches_found = 0
    total_singletons = 0

    for i in range(0, total_s1, batch_size):
        batch_s1 = s1_records[i : i + batch_size]
        batch_cands = blocker.retrieve_candidates(batch_s1, show_progress=False)

        batch_lookup = {
            r["entity_id"]: {
                "norm_name": normalize_business_name(r["business_name"]),
                "norm_addr": normalize_address(r["business_address"]),
            }
            for r in batch_s1
        }

        # Build feature matrix for batch candidate pairs
        features_list = []
        pairs_list = []
        for r in batch_s1:
            s1_id = r["entity_id"]
            s1_d = batch_lookup[s1_id]
            for cid in batch_cands.get(s1_id, []):
                cd = blocker.target_records.get(cid)
                if cd:
                    feat = extract_pair_features(
                        s1_norm_name=s1_d["norm_name"],
                        s1_norm_addr=s1_d["norm_addr"],
                        target_norm_name=cd["norm_name"],
                        target_norm_addr=cd["norm_addr"],
                        target_id=cid,
                    )
                    features_list.append(feat)
                    pairs_list.append((s1_id, cid))

        # Predict probabilities
        if features_list:
            X_batch = np.array(features_list, dtype=np.float32)
            probs = model.predict_proba(X_batch)
            matched_pairs_map = collections.defaultdict(list)
            for prob, (s1_id, cand_id) in zip(probs, pairs_list):
                if prob >= threshold:
                    matched_pairs_map[s1_id].append(cand_id)
        else:
            matched_pairs_map = collections.defaultdict(list)

        # Write results immediately to disk
        for r in batch_s1:
            s1_id = r["entity_id"]
            c_list = batch_cands.get(s1_id, [])
            m_list = matched_pairs_map.get(s1_id, [])
            # Deduplicate & ensure strict subset
            c_clean = list(dict.fromkeys(c_list))
            m_clean = [mid for mid in list(dict.fromkeys(m_list)) if mid in set(c_clean)]

            candidate_file_handle.write(f"{s1_id}\t{','.join(c_clean)}\n")
            matching_file_handle.write(f"{s1_id}\t{','.join(m_clean)}\n")

            if len(m_clean) == 0:
                total_singletons += 1
            else:
                total_matches_found += len(m_clean)

        processed = min(i + batch_size, total_s1)
        print(f"    Processed {processed:,}/{total_s1:,} {country} entities...")

    elapsed = time.time() - t0
    print(f"  Finished {country} in {elapsed/60:.2f} mins. Matches: {total_matches_found:,}, Singletons: {total_singletons:,}.")

    del blocker, s1_records
    gc.collect()


def generate_full_submission(
    train_dir: str = "student_resource/dataset/train",
    test_dir: str = "student_resource/dataset/test",
    output_dir: str = "output",
):
    os.makedirs(output_dir, exist_ok=True)
    matching_path = os.path.join(output_dir, "matching_results.tsv")
    candidate_path = os.path.join(output_dir, "candidate_pairs.tsv")

    print("=" * 80)
    print("AMAZON ML CHALLENGE 2026: HIGH-ACCURACY SUBMISSION GENERATOR")
    print(f"Output matching file:  {matching_path}")
    print(f"Output candidate file: {candidate_path}")
    print("=" * 80)

    # 1. Train model on training data
    train_s1, val_s1, val_gt, all_gt, train_targets = load_train_aligned_data(
        train_dir=train_dir,
        n_train_s1=15000,
        n_val_s1=1500,
    )

    model, threshold = train_matching_model(
        train_s1=train_s1,
        val_s1=val_s1,
        val_gt=val_gt,
        all_gt=all_gt,
        train_targets=train_targets,
    )
    del train_s1, val_s1, val_gt, all_gt, train_targets
    gc.collect()

    # 2. Run test inference streaming by country
    print("\n[4/5] Running Country-by-Country Inference on Full Test Set (1.73M entities)...")
    with open(candidate_path, "w", encoding="utf-8") as f_cand, open(matching_path, "w", encoding="utf-8") as f_match:
        # Write headers
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
        f_match.write("source1_entity_id\tmatched_entity_ids\n")

        # Process each country partition in order of size
        countries = ["France", "US", "India"]
        for ctry in countries:
            run_country_test_inference(
                country=ctry,
                test_dir=test_dir,
                model=model,
                threshold=threshold,
                candidate_file_handle=f_cand,
                matching_file_handle=f_match,
            )

    print("\n[5/5] Running Local Submission Validator...")
    val_script = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "student_resource", "utils", "validate_submission.py"
    )
    if os.path.isfile(val_script):
        cmd = [
            sys.executable,
            val_script,
            "--matching", matching_path,
            "--candidate", candidate_path,
            "--test-dir", test_dir,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
        if result.returncode == 0:
            print("\n>>> ALL VALIDATION CHECKS PASSED (exit code 0).")
            print(">>> The submission package is ready for portal upload!")
        else:
            print("\n>>> VALIDATION FAILED! Please inspect errors above.")
    else:
        print(f"Warning: Validator script not found at {val_script}")


if __name__ == "__main__":
    generate_full_submission()
