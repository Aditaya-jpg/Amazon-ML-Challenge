"""
High-speed pairwise feature engineering using RapidFuzz C++ routines.
Computes string similarity, token overlap, and address verification features.
"""

from typing import Dict, List, Optional, Tuple, Set
import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

from src.preprocess import extract_address_numbers


FEATURE_NAMES = [
    "name_levenshtein_ratio",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_jaro_winkler",
    "name_len_diff_ratio",
    "name_exact_match",
    "addr_levenshtein_ratio",
    "addr_token_sort_ratio",
    "addr_token_set_ratio",
    "addr_jaccard_similarity",
    "addr_number_match",
    "has_missing_addr",
    "target_is_s3",
    "interaction_name_addr",
]


def compute_jaccard(tokens1: set, tokens2: set) -> float:
    """Compute Jaccard similarity between two token sets."""
    if not tokens1 or not tokens2:
        return 0.0
    intersection = len(tokens1.intersection(tokens2))
    union = len(tokens1.union(tokens2))
    return float(intersection) / union if union > 0 else 0.0


def extract_pair_features(
    s1_norm_name: str,
    s1_norm_addr: str,
    target_norm_name: str,
    target_norm_addr: str,
    target_id: str,
) -> List[float]:
    """
    Extract vector of pairwise numerical features between S1 and target candidate.
    """
    # 1. Name features
    len_s1_name = len(s1_norm_name)
    len_tgt_name = len(target_norm_name)
    max_name_len = max(len_s1_name, len_tgt_name)
    len_diff = abs(len_s1_name - len_tgt_name) / (max_name_len + 1e-5)

    name_lev = fuzz.ratio(s1_norm_name, target_norm_name) / 100.0
    name_sort = fuzz.token_sort_ratio(s1_norm_name, target_norm_name) / 100.0
    name_set = fuzz.token_set_ratio(s1_norm_name, target_norm_name) / 100.0
    name_jw = JaroWinkler.similarity(s1_norm_name, target_norm_name)
    name_exact = 1.0 if s1_norm_name == target_norm_name and s1_norm_name != "" else 0.0

    # 2. Address features
    has_missing_addr = 1.0 if (not s1_norm_addr or not target_norm_addr) else 0.0

    if has_missing_addr:
        addr_lev = 0.0
        addr_sort = 0.0
        addr_set = 0.0
        addr_jaccard = 0.0
        addr_num_match = -1.0  # Flag indicating address numbers cannot be compared
    else:
        addr_lev = fuzz.ratio(s1_norm_addr, target_norm_addr) / 100.0
        addr_sort = fuzz.token_sort_ratio(s1_norm_addr, target_norm_addr) / 100.0
        addr_set = fuzz.token_set_ratio(s1_norm_addr, target_norm_addr) / 100.0
        
        s1_words = set(s1_norm_addr.split())
        tgt_words = set(target_norm_addr.split())
        addr_jaccard = compute_jaccard(s1_words, tgt_words)

        s1_nums = extract_address_numbers(s1_norm_addr)
        tgt_nums = extract_address_numbers(target_norm_addr)
        if s1_nums and tgt_nums:
            # Check if any common street/PIN number exists
            addr_num_match = 1.0 if (s1_nums & tgt_nums) else 0.0
        else:
            addr_num_match = -0.5

    # 3. Source & Interaction features
    target_is_s3 = 1.0 if target_id.startswith("S3-") else 0.0
    interaction_name_addr = name_sort * addr_sort

    return [
        name_lev,
        name_sort,
        name_set,
        name_jw,
        len_diff,
        name_exact,
        addr_lev,
        addr_sort,
        addr_set,
        addr_jaccard,
        addr_num_match,
        has_missing_addr,
        target_is_s3,
        interaction_name_addr,
    ]


def build_feature_matrix(
    candidate_map: Dict[str, List[str]],
    s1_records: Dict[str, Dict[str, str]],
    target_records: Dict[str, Dict[str, str]],
) -> Tuple[np.ndarray, List[Tuple[str, str]]]:
    """
    Build 2D feature matrix X for all (s1_id, target_id) candidate pairs.
    Returns (X, pair_ids) where pair_ids is a list of (s1_id, candidate_id).
    """
    features_list = []
    pair_ids = []

    for s1_id, cands in candidate_map.items():
        s1_data = s1_records.get(s1_id)
        if not s1_data:
            continue
        s1_name = s1_data["norm_name"]
        s1_addr = s1_data["norm_addr"]

        for cand_id in cands:
            cand_data = target_records.get(cand_id)
            if not cand_data:
                continue
            cand_name = cand_data["norm_name"]
            cand_addr = cand_data["norm_addr"]

            feat = extract_pair_features(
                s1_norm_name=s1_name,
                s1_norm_addr=s1_addr,
                target_norm_name=cand_name,
                target_norm_addr=cand_addr,
                target_id=cand_id,
            )
            features_list.append(feat)
            pair_ids.append((s1_id, cand_id))

    if not features_list:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32), []

    return np.array(features_list, dtype=np.float32), pair_ids
