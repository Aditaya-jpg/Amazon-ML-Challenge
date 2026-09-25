"""
High-recall, sub-linear multi-index blocking for business entity resolution.
Reduces O(N^2) comparison space to O(K * N) candidates (K <= 25).
"""

import collections
import re
from typing import Dict, List, Set, Tuple, Optional
from tqdm import tqdm

from src.preprocess import (
    normalize_business_name,
    normalize_address,
    extract_address_numbers
)

# Common frequent generic words across businesses that do not discriminate identity
STOP_TOKENS = {
    "the", "and", "for", "with", "all", "pro", "best", "new", "top",
    "services", "solutions", "enterprise", "enterprises", "consulting",
    "industries", "group", "holdings", "company", "center", "shop", "store",
    "tech", "technology", "technologies", "management", "international",
    "global", "national", "associates", "agency", "traders", "trading",
    "retail", "wholesale", "dealers", "distributors", "corp", "inc", "ltd"
}

RE_TOKEN = re.compile(r"\b[a-zA-Z0-9]{3,}\b")


def extract_informative_tokens(normalized_name: str) -> List[str]:
    """Extract distinct informative word tokens of length >= 3, excluding stop tokens."""
    if not normalized_name:
        return []
    tokens = RE_TOKEN.findall(normalized_name)
    return [t for t in set(tokens) if t not in STOP_TOKENS and not t.isdigit()]


def extract_name_shingles(normalized_name: str, n: int = 3) -> List[str]:
    """Extract character n-grams from business name (for typo tolerance)."""
    clean_str = "".join(normalized_name.split())
    if len(clean_str) < n:
        return [clean_str] if clean_str else []
    return [clean_str[i:i+n] for i in range(len(clean_str) - n + 1)]


def extract_address_blocking_keys(normalized_addr: str) -> List[str]:
    """
    Generate tight location blocking keys:
    e.g., street number + first 5 characters of first major address word.
    """
    if not normalized_addr:
        return []
    numbers = extract_address_numbers(normalized_addr)
    words = [w for w in normalized_addr.split() if not w.isdigit() and len(w) >= 4]
    if not numbers or not words:
        return []
    
    keys = []
    first_word = words[0][:5]
    for num in sorted(numbers):
        keys.append(f"{num}_{first_word}")
    return keys


class MultiIndexBlocker:
    """
    Inverted index-based blocker operating per country partition.
    Combines:
    1. Rare Token Inverted Index
    2. Character 3-Gram Inverted Index
    3. Location/Street Number Inverted Index
    """

    def __init__(self, max_df_token: int = 3500, max_candidates_per_entity: int = 25):
        self.max_df_token = max_df_token
        self.max_candidates = max_candidates_per_entity
        self.token_index = collections.defaultdict(list)
        self.shingle_index = collections.defaultdict(list)
        self.addr_key_index = collections.defaultdict(list)
        self.target_records = {}  # entity_id -> {norm_name, norm_addr}

    def index_targets(self, target_data: List[Dict[str, str]], show_progress: bool = False):
        """
        Build inverted indices over target records (Source 2 and Source 3).
        target_data expects dicts with keys: entity_id, business_name, business_address.
        """
        self.token_index.clear()
        self.shingle_index.clear()
        self.addr_key_index.clear()
        self.target_records.clear()

        # Step 1: Preprocess and count token frequencies
        token_doc_counts = collections.Counter()
        shingle_doc_counts = collections.Counter()

        iterator = tqdm(target_data, desc="Indexing targets") if show_progress else target_data
        
        # Temporary storage of parsed tokens
        parsed_entries = []
        for rec in iterator:
            eid = rec["entity_id"]
            norm_name = normalize_business_name(rec.get("business_name", ""))
            norm_addr = normalize_address(rec.get("business_address", ""))
            self.target_records[eid] = {
                "norm_name": norm_name,
                "norm_addr": norm_addr,
                "raw_name": rec.get("business_name", ""),
                "raw_addr": rec.get("business_address", "")
            }

            tokens = extract_informative_tokens(norm_name)
            shingles = extract_name_shingles(norm_name, n=3)
            addr_keys = extract_address_blocking_keys(norm_addr)

            token_doc_counts.update(tokens)
            shingle_doc_counts.update(shingles)
            parsed_entries.append((eid, tokens, shingles, addr_keys))

        # Step 2: Populate inverted index with tokens within selectivity range
        for eid, tokens, shingles, addr_keys in parsed_entries:
            for t in tokens:
                # Discard overly common terms to preserve selectivity
                if token_doc_counts[t] <= self.max_df_token:
                    self.token_index[t].append(eid)
            
            # Selectively index rare 3-grams
            for sh in shingles:
                if 2 <= shingle_doc_counts[sh] <= 800:
                    self.shingle_index[sh].append(eid)

            # Index address keys
            for ak in addr_keys:
                self.addr_key_index[ak].append(eid)

    def retrieve_candidates(
        self,
        ref_records: List[Dict[str, str]],
        show_progress: bool = False
    ) -> Dict[str, List[str]]:
        """
        Query inverted indices for each reference record (Source 1).
        Returns a mapping: source1_entity_id -> list of candidate entity_ids.
        """
        results = {}
        iterator = tqdm(ref_records, desc="Retrieving candidates") if show_progress else ref_records

        for rec in iterator:
            s1_id = rec["entity_id"]
            norm_name = normalize_business_name(rec.get("business_name", ""))
            norm_addr = normalize_address(rec.get("business_address", ""))

            s1_tokens = extract_informative_tokens(norm_name)
            s1_shingles = extract_name_shingles(norm_name, n=3)
            s1_addr_keys = extract_address_blocking_keys(norm_addr)

            # Candidate scoring accumulator: candidate_id -> score
            candidate_scores = collections.defaultdict(float)

            # Query Pass 1: Rare Name Tokens (High weight)
            for t in s1_tokens:
                matches = self.token_index.get(t, [])
                if matches:
                    weight = 1.0 + 1.0 / (len(matches) ** 0.5)
                    for mid in matches:
                        candidate_scores[mid] += weight

            # Query Pass 2: Character 3-Grams (Typo bridge)
            for sh in s1_shingles:
                matches = self.shingle_index.get(sh, [])
                if matches:
                    weight = 0.2 / (len(matches) ** 0.5)
                    for mid in matches:
                        candidate_scores[mid] += weight

            # Query Pass 3: Location / Address Key (DBA bridge)
            for ak in s1_addr_keys:
                matches = self.addr_key_index.get(ak, [])
                if matches:
                    weight = 1.5 + 1.0 / (len(matches) ** 0.5)
                    for mid in matches:
                        candidate_scores[mid] += weight

            if not candidate_scores:
                results[s1_id] = []
            else:
                # Sort by accumulated blocking score and take top K
                top_candidates = sorted(
                    candidate_scores.keys(),
                    key=lambda k: candidate_scores[k],
                    reverse=True
                )[:self.max_candidates]
                results[s1_id] = top_candidates

        return results
