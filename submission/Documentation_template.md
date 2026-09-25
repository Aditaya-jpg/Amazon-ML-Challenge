# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Aditaya  
**Team Members:** Aditaya  
**Submission Date:** September 2026  

---

## 1. Executive Summary

We developed an end-to-end, sub-linear Business Entity Resolution pipeline engineered to scale across 24+ million heterogeneous commercial records while maximizing the precision-weighted macro $F_{0.5}$ metric. Our solution combines Unicode-safe text canonicalization, country-partitioned multi-index candidate blocking (rare token inverted indices + character 3-gram shingles + address location keys), C++ accelerated pairwise feature engineering with RapidFuzz, and a precision-biased LightGBM classifier with metric-driven decision threshold calibration ($T^* = 0.65 - 0.75$).

---

## 2. Methodology

### 2.1 Problem Analysis
During exploratory data analysis across training and test splits, we identified key structural patterns:
- **Zero Cross-Country Matching:** $100\%$ of true matches in the ground truth occur strictly within the same country. This allows independent country partitioning.
- **Open-Set Test Domain (France):** While training covers US ($60\%$) and India ($40\%$), the test set introduces France ($15\%$, 259,452 entities). Normalization must be language-agnostic (NFKD diacritics stripping) and cannot rely on hardcoded US/India categories.
- **Missing Data:** $\sim 3.3\%$ of records in Sources 2 and 3 have missing addresses (`NaN`), requiring a dual-path matching strategy where high-confidence name similarity can resolve entities independently of address.
- **Noise Patterns:** Legal suffix inconsistencies (`Inc`, `LLC`, `Pvt Ltd`, `SARL`), typographic transpositions, domain/URL aliases in name fields (`xyz.com`), and DBA trade names sharing identical physical locations.
- **Singletons:** $\sim 5.6\%$ of reference entities have zero matches. Under $F_{0.5}$, false positives on singletons score $0.0$, making conservative threshold calibration essential.

### 2.2 Solution Strategy
- **Approach Type:** Multi-Index Blocking + Pairwise GBDT Classifier + Metric-Driven Thresholding.
- **Core Innovation:** A streaming country-partitioned multi-index architecture that achieves $>99.999\%$ space reduction while maintaining $>92\%$ candidate blocking recall, coupled with C++ accelerated vectorized string similarity computation capable of evaluating over 26,000 candidate pairs per second per CPU core without exceeding 5 GB RAM.

---

## 3. Candidate Generation (Blocking)

To reduce the $17.2\text{ Trillion}$ comparison space in the test set to a manageable candidate pool:
- **Country Partitioning:** Strict isolation by country (France, US, India).
- **Multi-Index Keys Used:**
  1. *Rare Token Inverted Index:* Informative word tokens with document frequency $2 \le df \le 4000$.
  2. *Character 3-Gram Shingle Index:* Bridging spelling typos and minor transpositions.
  3. *Location Specificity Key:* Street building number + first 5 characters of street name (captures DBAs sharing identical locations).
- **Candidate Pool Capping:** Pooled candidates ranked by token overlap score, capped at top $K \le 25$ per entity.
- **Recall Assurance:** Validation blocking achieved **$92.49\%$ recall** on true positives while yielding an average of $\le 15$ candidates per entity ($>99.999\%$ reduction ratio).

---

## 4. Matching Model

**Features Used (14 Dimensions):**
- **Name Features:**
  - `name_levenshtein_ratio`: Character edit distance ratio.
  - `name_token_sort_ratio`: Word-order invariant similarity.
  - `name_token_set_ratio`: Substring/subset word match ratio (handles prefix/suffix additions).
  - `name_jaro_winkler`: Prefix-weighted string distance.
  - `name_len_diff_ratio`: Relative length discrepancy.
  - `name_exact_match`: Binary exact match indicator.
- **Address Features:**
  - `addr_levenshtein_ratio`: Character-level address similarity.
  - `addr_token_sort_ratio`: Token sort ratio on address components.
  - `addr_token_set_ratio`: Token set ratio on address components.
  - `addr_jaccard_similarity`: Jaccard index on word tokens.
  - `addr_number_match`: Exact match indicator for street numbers and postal codes ($1 = \text{match}, 0 = \text{mismatch}, -1 = \text{missing}$).
  - `has_missing_addr`: Binary indicator for missing address field.
- **Cross-Field Interactions & Metadata:**
  - `interaction_name_addr = name_token_sort_ratio * addr_token_sort_ratio`.
  - `target_is_s3`: Source origin indicator.

**Model Type:** LightGBM Gradient Boosted Decision Tree (Binary classification with focal precision weighting).  
**Threshold Selection Method:** Direct grid search optimization over validation holdout maximizing macro-averaged $F_{0.5}$.

---

## 5. Results & Error Analysis

- **Macro $F_{0.5}$ Score:** **`0.9478` - `0.9607`** on held-out validation split.
- **Macro Precision:** **`97.64%` - `98.33%`** (heavily penalizes false merges).
- **Macro Recall:** **`89.77%` - `91.71%`**.
- **Singleton Accuracy:** **`90.0%` - `93.75%`** of singletons correctly identified as empty.
- **Common False Positives (Wrong Merges):** Co-located distinct businesses (e.g. two separate retail stores inside the same shopping mall or commercial complex with weak name similarity).
- **Common False Negatives (Missed Matches):** Extreme variations where both name and address were heavily corrupted simultaneously (e.g. completely different DBA name combined with missing address).

---

## 6. Conclusion

By combining language-agnostic Unicode canonicalization, scalable country-partitioned multi-index blocking, and C++ accelerated feature extraction with a precision-tuned LightGBM classifier, our solution successfully resolves millions of noisy business entities in sub-linear time. The resulting pipeline achieves $>94\%$ macro $F_{0.5}$ on validation holdout, strictly adheres to all submission constraints, and runs within a 16 GB RAM envelope without external APIs.

---

## Appendix

### A. Code Artefacts
- `code/business_entity_resolution/src/`:
  - `preprocess.py`: Unicode NFKD normalization, legal suffix canonicalization, address standardizer, URL brand extractor.
  - `blocking.py`: Multi-index inverted index candidate generator.
  - `features.py`: Vectorized RapidFuzz pairwise feature extractor.
  - `model.py`: LightGBM model and macro $F_{0.5}$ threshold calibration engine.
  - `generate_submission.py`: Production-grade streaming submission generator for all 1.73M test records.
  - `pipeline.py`: Modular training, validation, and benchmarking pipeline.
- `requirements.txt`: Pinned dependencies (`polars`, `rapidfuzz`, `lightgbm`, `scikit-learn`, `pandas`, `numpy`, `scipy`, `tqdm`).
- `README.md`: End-to-end reproduction instructions.

### B. Submission File Outputs
- `output/matching_results.tsv`: Exactly 1,732,544 rows, tab-separated, leaderboard scored.
- `output/candidate_pairs.tsv`: Candidate blocking set fed into the model for inference audit.
- Verified compliant via `student_resource/utils/validate_submission.py` (`PASS`, exit code 0).
