"""
End-to-end Pipeline: Training, Validation, and Test Inference.
Outputs matching_results.tsv and candidate_pairs.tsv adhering to competition format.
"""

import argparse
import os
import sys
import collections
import subprocess
from typing import Dict, List, Set, Tuple, Optional
import pandas as pd
import numpy as np

# Ensure src modules can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess import normalize_business_name, normalize_address
from src.blocking import MultiIndexBlocker
from src.features import build_feature_matrix
from src.model import EntityMatcherModel, evaluate_macro_f05


def load_ground_truth(path: str, nrows: Optional[int] = None) -> Dict[str, Set[str]]:
    """Load ground truth mapping: source1_entity_id -> set of matched entity IDs."""
    print(f"Loading Ground Truth from {path}...")
    df = pd.read_csv(path, sep="\t", nrows=nrows)
    gt = {}
    for _, row in df.iterrows():
        s1 = str(row["source1_entity_id"]).strip()
        m_str = str(row["matched_entity_ids"]).strip() if pd.notna(row["matched_entity_ids"]) else ""
        if not m_str or m_str.lower() == "nan":
            gt[s1] = set()
        else:
            gt[s1] = set(m.strip() for m in m_str.split(",") if m.strip())
    return gt


def load_tsv_records(path: str, needed_ids: Optional[Set[str]] = None, max_sample_rows: Optional[int] = None) -> List[Dict[str, str]]:
    """
    Load TSV records efficiently.
    If needed_ids is given, ensures all needed IDs are captured by streaming through chunks,
    plus up to max_sample_rows additional rows for candidate negatives.
    """
    records = []
    found_needed = set()
    total_needed = len(needed_ids) if needed_ids else 0
    sample_budget = max_sample_rows if max_sample_rows is not None else float("inf")

    chunk_size = 150000
    for chunk in pd.read_csv(path, sep="\t", chunksize=chunk_size):
        for col in chunk.columns:
            chunk[col] = chunk[col].fillna("").astype(str)

        chunk_recs = chunk.to_dict(orient="records")
        for r in chunk_recs:
            eid = r["entity_id"]
            if needed_ids and eid in needed_ids:
                records.append(r)
                found_needed.add(eid)
            elif sample_budget > 0:
                records.append(r)
                sample_budget -= 1

        if (needed_ids is None or len(found_needed) >= total_needed) and sample_budget <= 0:
            break

    print(f"Loaded {len(records):,} records from {os.path.basename(path)} (captured {len(found_needed)}/{total_needed} target positives).")
    return records


def find_validator_script() -> Optional[str]:
    """Locate validate_submission.py in known project directories."""
    candidates = [
        os.path.join("student_resource", "utils", "validate_submission.py"),
        os.path.join("..", "student_resource", "utils", "validate_submission.py"),
        os.path.join("..", "..", "student_resource", "utils", "validate_submission.py"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "student_resource", "utils", "validate_submission.py"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return os.path.abspath(c)
    return None


def run_pipeline(
    train_dir: str,
    test_dir: str,
    output_dir: str,
    sample_size: Optional[int] = None,
    eval_only: bool = False,
):
    os.makedirs(output_dir, exist_ok=True)
    matching_out = os.path.join(output_dir, "matching_results.tsv")
    candidate_out = os.path.join(output_dir, "candidate_pairs.tsv")

    print("=" * 80)
    print("AMAZON ML CHALLENGE 2026: BUSINESS ENTITY RESOLUTION PIPELINE")
    print(f"Train dir:   {train_dir}")
    print(f"Test dir:    {test_dir}")
    print(f"Output:      {output_dir}")
    print(f"Sample size: {sample_size}")
    print("=" * 80)

    # -------------------------------------------------------------
    # STEP 1: LOAD SOURCE 1 RECORDS & ALIGNED GROUND TRUTH
    # -------------------------------------------------------------
    s1_all_df = pd.read_csv(os.path.join(train_dir, "train_source1.tsv"), sep="\t", nrows=sample_size)
    for col in s1_all_df.columns:
        s1_all_df[col] = s1_all_df[col].fillna("").astype(str)
    s1_all = s1_all_df.to_dict(orient="records")
    s1_needed = set(s1_all_df["entity_id"])

    print(f"Loaded {len(s1_all):,} Source 1 reference records. Aligning Ground Truth...")
    gt_all = {}
    for chunk in pd.read_csv(os.path.join(train_dir, "train_ground_truth.tsv"), sep="\t", chunksize=150000):
        sub = chunk[chunk["source1_entity_id"].isin(s1_needed)]
        for _, r in sub.iterrows():
            s1 = r["source1_entity_id"]
            m_str = str(r["matched_entity_ids"]).strip() if pd.notna(r["matched_entity_ids"]) else ""
            if m_str and m_str.lower() != "nan":
                gt_all[s1] = set(m.strip() for m in m_str.split(",") if m.strip())
            else:
                gt_all[s1] = set()
        if len(gt_all) >= len(s1_needed):
            break

    # Stratified Train/Val split on S1 entities (90% Train, 10% Val)
    np.random.seed(42)
    s1_ids = [r["entity_id"] for r in s1_all]
    np.random.shuffle(s1_ids)

    n_val = max(1, int(len(s1_ids) * 0.10))
    val_s1_id_set = set(s1_ids[:n_val])
    train_s1_id_set = set(s1_ids[n_val:])

    train_s1 = [r for r in s1_all if r["entity_id"] in train_s1_id_set]
    val_s1 = [r for r in s1_all if r["entity_id"] in val_s1_id_set]
    val_gt = {eid: gt_all.get(eid, set()) for eid in val_s1_id_set}

    print(f"Split summary: {len(train_s1):,} Train S1, {len(val_s1):,} Validation S1 entities.")

    # -------------------------------------------------------------
    # STEP 2: LOAD TARGET SOURCES (S2 & S3) FOR TRAINING
    # -------------------------------------------------------------
    # Collect all positive target IDs required for train and val S1 records
    needed_target_ids = set()
    for eid in s1_ids:
        needed_target_ids.update(gt_all.get(eid, set()))

    print(f"Required target positive IDs for ground truth: {len(needed_target_ids):,}")

    max_neg_sample = (sample_size * 4) if sample_size is not None else None
    train_s2 = load_tsv_records(os.path.join(train_dir, "train_source2.tsv"), needed_ids=needed_target_ids, max_sample_rows=max_neg_sample)
    train_s3 = load_tsv_records(os.path.join(train_dir, "train_source3.tsv"), needed_ids=needed_target_ids, max_sample_rows=max_neg_sample)
    train_targets_all = train_s2 + train_s3
    del train_s2, train_s3

    # Group records by country partition
    ctry_train_s1 = collections.defaultdict(list)
    for r in train_s1:
        ctry_train_s1[r["country"]].append(r)

    ctry_val_s1 = collections.defaultdict(list)
    for r in val_s1:
        ctry_val_s1[r["country"]].append(r)

    ctry_targets = collections.defaultdict(list)
    for r in train_targets_all:
        ctry_targets[r["country"]].append(r)
    del train_targets_all

    # -------------------------------------------------------------
    # STEP 3: CANDIDATE GENERATION & FEATURE EXTRACTION (PER COUNTRY)
    # -------------------------------------------------------------
    X_train_list, y_train_list = [], []
    X_val_list = []
    val_pair_ids = []

    blocker = MultiIndexBlocker(max_df_token=3500, max_candidates_per_entity=25)

    for country in sorted(ctry_targets.keys()):
        targets = ctry_targets[country]
        tr_s1 = ctry_train_s1[country]
        vl_s1 = ctry_val_s1[country]

        print(f"\n--- Processing Country Partition: {country} ---")
        print(f"Targets: {len(targets):,}, Train S1: {len(tr_s1):,}, Val S1: {len(vl_s1):,}")

        # Index target records
        blocker.index_targets(targets, show_progress=False)

        # 3a. Retrieve candidates for Train S1
        tr_cands = blocker.retrieve_candidates(tr_s1, show_progress=False)

        # Inject true matches into training candidates so the model learns positive patterns
        for r in tr_s1:
            eid = r["entity_id"]
            true_pos = gt_all.get(eid, set())
            current_cands = set(tr_cands.get(eid, []))
            for tp in true_pos:
                if tp in blocker.target_records and tp not in current_cands:
                    tr_cands[eid].append(tp)

        s1_lookup = {}
        for r in tr_s1:
            s1_lookup[r["entity_id"]] = {
                "norm_name": normalize_business_name(r.get("business_name", "")),
                "norm_addr": normalize_address(r.get("business_address", ""))
            }

        X_tr_part, tr_pairs = build_feature_matrix(tr_cands, s1_lookup, blocker.target_records)
        y_tr_part = np.array([1 if p[1] in gt_all.get(p[0], set()) else 0 for p in tr_pairs], dtype=np.int32)

        X_train_list.append(X_tr_part)
        y_train_list.append(y_tr_part)

        # 3b. Retrieve candidates for Validation S1 (Strictly no ground truth injection)
        vl_cands = blocker.retrieve_candidates(vl_s1, show_progress=False)
        vl_s1_lookup = {}
        for r in vl_s1:
            vl_s1_lookup[r["entity_id"]] = {
                "norm_name": normalize_business_name(r.get("business_name", "")),
                "norm_addr": normalize_address(r.get("business_address", ""))
            }

        X_vl_part, vl_pairs = build_feature_matrix(vl_cands, vl_s1_lookup, blocker.target_records)
        X_val_list.append(X_vl_part)
        val_pair_ids.extend(vl_pairs)

    X_train = np.vstack(X_train_list) if X_train_list else np.empty((0, 14))
    y_train = np.concatenate(y_train_list) if y_train_list else np.empty((0,))
    X_val = np.vstack(X_val_list) if X_val_list else np.empty((0, 14))

    print(f"\nFinal Feature Matrices: X_train shape={X_train.shape}, X_val shape={X_val.shape}")
    print(f"X_train positives: {np.sum(y_train):,} / {len(y_train):,} ({np.mean(y_train)*100:.2f}%)")

    # -------------------------------------------------------------
    # STEP 4: TRAIN MODEL & OPTIMIZE THRESHOLD
    # -------------------------------------------------------------
    matcher = EntityMatcherModel(threshold=0.75)
    matcher.fit(X_train, y_train)

    # Tune decision threshold to maximize macro F_0.5 on validation holdout
    best_threshold = matcher.optimize_threshold(
        X_val,
        val_pair_ids,
        val_gt,
        threshold_candidates=[0.40, 0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
    )

    # Detailed metrics computation on validation holdout
    val_probs = matcher.predict_proba(X_val)
    val_preds_map = collections.defaultdict(set)
    for p, (s1, cand) in zip(val_probs, val_pair_ids):
        if p >= best_threshold:
            val_preds_map[s1].add(cand)

    val_f05_list = []
    val_precisions = []
    val_recalls = []
    total_singletons = 0
    correct_singletons = 0

    for s1_id, true_set in val_gt.items():
        pred_set = val_preds_map.get(s1_id, set())
        from src.model import compute_entity_f05
        f05 = compute_entity_f05(true_set, pred_set)
        val_f05_list.append(f05)

        if len(true_set) == 0:
            total_singletons += 1
            if len(pred_set) == 0:
                correct_singletons += 1
        else:
            if len(pred_set) > 0:
                tp = len(true_set.intersection(pred_set))
                val_precisions.append(tp / len(pred_set))
                val_recalls.append(tp / len(true_set))
            else:
                val_precisions.append(0.0)
                val_recalls.append(0.0)

    print("\n" + "=" * 80)
    print("OFFICIAL COMPETITION EVALUATION METRICS (VALIDATION HOLDOUT)")
    print("=" * 80)
    print(f"  Macro F_0.5 Score:             {np.mean(val_f05_list):.4f}")
    print(f"  Macro Precision (non-sing):    {np.mean(val_precisions):.4f} ({np.mean(val_precisions)*100:.2f}%)")
    print(f"  Macro Recall (non-sing):       {np.mean(val_recalls):.4f} ({np.mean(val_recalls)*100:.2f}%)")
    print(f"  Singleton Accuracy:            {correct_singletons}/{max(1, total_singletons)} ({correct_singletons/max(1, total_singletons)*100:.2f}%)")
    print(f"  Total Validation Entities:     {len(val_gt):,}")
    print("=" * 80)

    if eval_only:
        print("\nEval-only mode requested. Skipping test inference.")
        return

    # -------------------------------------------------------------
    # STEP 5: TEST SET INFERENCE
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 5: RUNNING INFERENCE ON TEST SET")
    print("=" * 80)

    test_s1_df = pd.read_csv(os.path.join(test_dir, "test_source1.tsv"), sep="\t", nrows=sample_size)
    for col in test_s1_df.columns:
        test_s1_df[col] = test_s1_df[col].fillna("").astype(str)
    test_s1 = test_s1_df.to_dict(orient="records")

    target_test_sample = (sample_size * 5) if sample_size is not None else None
    test_s2 = load_tsv_records(os.path.join(test_dir, "test_source2.tsv"), max_sample_rows=target_test_sample)
    test_s3 = load_tsv_records(os.path.join(test_dir, "test_source3.tsv"), max_sample_rows=target_test_sample)
    test_targets_all = test_s2 + test_s3
    del test_s2, test_s3

    # Group test records by country partition (US, India, France!)
    ctry_test_s1 = collections.defaultdict(list)
    for r in test_s1:
        ctry_test_s1[r["country"]].append(r)

    ctry_test_targets = collections.defaultdict(list)
    for r in test_targets_all:
        ctry_test_targets[r["country"]].append(r)
    del test_targets_all

    final_candidates = {}
    final_matches = {}

    for country in sorted(ctry_test_s1.keys()):
        s1_records = ctry_test_s1[country]
        target_records = ctry_test_targets.get(country, [])

        print(f"\nInferencing Country: {country} (S1: {len(s1_records):,}, Targets: {len(target_records):,})")

        blocker.index_targets(target_records, show_progress=False)
        cands = blocker.retrieve_candidates(s1_records, show_progress=False)

        s1_lookup = {}
        for r in s1_records:
            s1_lookup[r["entity_id"]] = {
                "norm_name": normalize_business_name(r.get("business_name", "")),
                "norm_addr": normalize_address(r.get("business_address", ""))
            }

        X_test_part, test_pairs = build_feature_matrix(cands, s1_lookup, blocker.target_records)
        probs = matcher.predict_proba(X_test_part)

        s1_matches_map = collections.defaultdict(list)
        for prob, (s1_id, cand_id) in zip(probs, test_pairs):
            if prob >= best_threshold:
                s1_matches_map[s1_id].append(cand_id)

        for r in s1_records:
            s1_id = r["entity_id"]
            c_list = cands.get(s1_id, [])
            m_list = s1_matches_map.get(s1_id, [])
            final_candidates[s1_id] = c_list
            # Ensure final matches are a strict subset of candidate pairs
            final_matches[s1_id] = [mid for mid in m_list if mid in set(c_list)]

    # -------------------------------------------------------------
    # STEP 6: EXPORT SUBMISSION TSV FILES
    # -------------------------------------------------------------
    # Load full list of test S1 IDs to guarantee all S1 entities appear in output
    print(f"\nEnsuring all test S1 entities are present in final submission files...")
    all_test_s1_df = pd.read_csv(os.path.join(test_dir, "test_source1.tsv"), sep="\t", usecols=["entity_id"])
    all_test_s1_ids = all_test_s1_df["entity_id"].tolist()

    print(f"Writing {candidate_out} ({len(all_test_s1_ids):,} total rows)...")
    with open(candidate_out, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in all_test_s1_ids:
            cands = final_candidates.get(s1_id, [])
            f.write(f"{s1_id}\t{','.join(cands)}\n")

    print(f"Writing {matching_out} ({len(all_test_s1_ids):,} total rows)...")
    with open(matching_out, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in all_test_s1_ids:
            matches = final_matches.get(s1_id, [])
            f.write(f"{s1_id}\t{','.join(matches)}\n")

    print(f"\nOutputs written successfully:")
    print(f"  matching_results.tsv: {len(all_test_s1_ids):,} rows")
    print(f"  candidate_pairs.tsv:  {len(all_test_s1_ids):,} rows")

    # -------------------------------------------------------------
    # STEP 7: RUN VALIDATOR SCRIPT
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 7: RUNNING LOCAL SUBMISSION VALIDATION")
    print("=" * 80)
    val_script = find_validator_script()
    if val_script:
        cmd = [
            sys.executable,
            val_script,
            "--matching", matching_out,
            "--candidate", candidate_out,
            "--test-dir", test_dir
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
        if result.returncode == 0:
            print("Validation PASSED! Files are fully compliant.")
        else:
            print("Validation FAILED! Check error messages above.")
    else:
        print("Warning: Validator script not found!")


def main():
    parser = argparse.ArgumentParser(description="End-to-End Business Entity Resolution Pipeline")
    parser.add_argument("--train-dir", default="student_resource/dataset/train", help="Path to train dataset directory")
    parser.add_argument("--test-dir", default="student_resource/dataset/test", help="Path to test dataset directory")
    parser.add_argument("--output-dir", default="output", help="Directory to save output files")
    parser.add_argument("--sample-size", type=int, default=None, help="Sample size for fast validation runs (default: None for full run)")
    parser.add_argument("--eval-only", action="store_true", help="Only evaluate on validation holdout, skip test inference")
    args = parser.parse_args()

    run_pipeline(
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        sample_size=args.sample_size,
        eval_only=args.eval_only,
    )


if __name__ == "__main__":
    main()
