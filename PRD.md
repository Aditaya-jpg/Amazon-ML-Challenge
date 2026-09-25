# Product Requirements Document (PRD)
## Business Entity Resolution System — Amazon ML Challenge 2026

---

### 1. Document Control & Metadata
- **Project Name:** Business Entity Resolution System (ER-3S)
- **Competition:** Amazon ML Challenge 2026
- **Repository:** [Aditaya-jpg/Amazon-ML-Challenge](https://github.com/Aditaya-jpg/Amazon-ML-Challenge)
- **Target Metric:** Macro-averaged $F_{0.5}$ Score (Precision-Weighted)
- **Date:** September 2026
- **Status:** Approved / Architecture Design Phase

---

### 2. Executive Summary & Objective

In modern e-commerce and commercial ecosystems, seller and business records arrive asynchronously from diverse external sources (merchant registries, tax records, logistics partners, supply chain databases). These data sources contain noisy fragments: missing fields, legal suffix discrepancies, transliteration variants, abbreviations, address permutations, and even website URLs in place of business names.

The objective of this project is to construct an end-to-end, reproducible Machine Learning pipeline that maps business entities from **Source 1** (the reference ground truth deduplicated source) to their corresponding identity records in **Source 2** and **Source 3**. 

Each Source 1 entity may link to zero (singleton), one, or multiple records across Source 2 and Source 3. The system must operate without any external API lookups or web augmentation, scale to over 11.7 million records in inference under constrained memory (16 GB RAM), and maximize the macro-averaged $F_{0.5}$ score.

---

### 3. Problem Scope & Data Characteristics

#### 3.1 Data Architecture
| Dataset Split | Source 1 (Reference) | Source 2 (Target) | Source 3 (Target) | Total Records |
| :--- | :--- | :--- | :--- | :--- |
| **Train** | 2,206,821 | 5,034,616 | 5,285,603 | **12,527,040** |
| **Test** | 1,732,544 | 4,887,273 | 5,082,316 | **11,702,133** |
| **Total** | 3,939,365 | 9,921,889 | 10,367,919 | **24,229,173** |

#### 3.2 Schema
Each source table (`*_source1.tsv`, `*_source2.tsv`, `*_source3.tsv`) shares a standard TSV schema:
- `entity_id` *(string, PK)*: Source prefixed identifier (`S1-xxxx`, `S2-xxxx`, `S3-xxxx`).
- `business_name` *(string)*: Registered or commercial name.
- `business_address` *(string)*: Address string with variable component ordering and quality.
- `country` *(string)*: Country ISO/Name label.

Ground Truth (`train_ground_truth.tsv`):
- `source1_entity_id` *(string, PK)*: S1 reference ID.
- `matched_entity_ids` *(string)*: Comma-separated list of true S2/S3 IDs (`""` if singleton).

#### 3.3 Noise Patterns Discovered via EDA
1. **Missing Data Patterns:**
   - Business names: Almost 100% populated (< 0.001% missing).
   - Business addresses: ~3.3% missing in Source 2 and Source 3. When address is `NaN`, resolution must rely exclusively on high-confidence name/phonetic/brand matching.
2. **Name Variations:**
   - Legal suffix expansions/drops: `Inc`, `LLP`, `Corp`, `Private Limited`, `Pvt Ltd`, `SARL`, `SAS`, `SA`.
   - Typographic errors: Character deletions/transpositions (`Maure Wilblims` vs `Maure Williams`).
   - Domain/URL aliases: Entity name recorded as URL (`maurewilliamscolombier.com`).
   - DBA/Parent trade names: Completely different names (e.g. `Dröxkor` vs `Maure Williams Colombier Inc`) linked to identical physical addresses.
3. **Address Variations:**
   - Abbreviations: `Ave` vs `Avenue`, `Rd` vs `Road`, `St` vs `Street`.
   - Country/State syntax: `NY` vs `New York`, `NC` vs `North Carolina`.
   - Structural re-ordering: `IA, Iowa City, 1064 Newton Rd` vs `1064 Newton Rd, Unit 11, Iowa City, IA`.
   - Landmark-centric addresses (India): e.g. `Near SBI ATM, Sector 14`.
4. **Country Partitioning & Open-Set Reality:**
   - **Empirical Validation:** 0% cross-country matching found in ground truth. Entities strictly resolve within their respective country.
   - **Training Set Countries:** US (~60%), India (~40%).
   - **Test Set Countries:** India (~46.8%), US (~38.3%), France (~15.0%).
   - **Requirement:** Pipeline cannot hardcode country lists or categorical one-hot vectors. France (unseen in training) must be handled gracefully through generalized text normalization and intra-country blocking.

---

### 4. Technical Constraints & Compliance

1. **Anti-Leakage & Integrity Policy:**
   - External databases, commercial APIs, geocoding lookups, and internet queries are **strictly prohibited**. Disqualification is automated and non-negotiable.
2. **Model Restrictions:**
   - MIT or Apache 2.0 open-source license only.
   - Parameter ceiling: $\le 8$ Billion parameters.
3. **Hardware & Latency Constraints:**
   - Environment: 16 GB Physical RAM, CPU execution.
   - Brute-force complexity: $O(|S_1| \times (|S_2| + |S_3|)) \approx 1.72 \times 10^{13}$ comparisons is completely intractable.
   - Inference Memory Budget: Peak memory footprint must stay $\le 10$ GB to prevent OS thrashing/OOM. Chunked streaming processing is mandatory.
4. **Submission Format Integrity:**
   - File 1: `matching_results.tsv` (Leaderboard scored). Header: `source1_entity_id\tmatched_entity_ids`.
   - File 2: `candidate_pairs.tsv` (Validation & audit). Header: `source1_entity_id\tcandidate_entity_ids`.
   - Every single $S_1$ entity from test set must appear exactly once.
   - Matched IDs must be a strict subset of Candidate IDs.
   - All outputs validated via `validate_submission.py`.

---

### 5. Evaluation Metric Specification

The challenge evaluates using macro-averaged $F_{\beta}$ with $\beta = 0.5$:
$$F_{0.5} = \frac{(1 + 0.5^2) \times \text{Precision} \times \text{Recall}}{0.5^2 \times \text{Precision} + \text{Recall}} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$

#### Metric Dynamics:
- **Precision Weighting:** Precision is weighted $2\times$ as heavily as Recall. A false positive (merging unrelated entities) destroys score twice as fast as a missed link.
- **Singleton Score Impact:** Singletons (~5.6% of test set, ~97,000 entities) yield $1.0$ if predicted empty, but immediately drop to $0.0$ if even a single candidate is falsely accepted.
- **Decision Boundary Principle:** Matching threshold $T$ must be tuned aggressively towards high precision ($T \approx 0.65 - 0.85$), rejecting marginal candidates.

---

### 6. Architectural Requirements & Non-Functional Goals

- **Modular Stages:** Independent blocking, feature extraction, scoring, thresholding, and verification modules.
- **Scalability:** Must process 1.73M reference entities and 9.96M target entities in $\le 3$ hours on CPU.
- **Determinism & Reproducibility:** Fixed seeds, zero random variation, clean script execution.
- **Strict Separation of Concerns:**
  - `code/business_entity_resolution/src/`: Core logic.
  - `output/`: TSV submission artifacts.
  - `Documentation_template.md`: Methodological report.

---

### 7. Success Criteria & KPIs

| Phase | Milestone / KPI | Target |
| :--- | :--- | :--- |
| **Blocking Recall** | Candidate Generation Recall Upper Bound | $\ge 92.0\%$ true match recall |
| **Reduction Ratio** | Space Reduction vs Brute Force | $\ge 99.995\%$ (Avg $\le 25$ candidates per $S_1$) |
| **Validation Score** | Macro $F_{0.5}$ on 10% Stratified Validation Holdout | $\ge 0.8500$ |
| **Format Audit** | `validate_submission.py` Execution | Clean `PASS` (0 errors, 0 invalid IDs) |
| **Runtime** | Full Test Inference Pipeline Duration | $< 180$ minutes on 16 GB CPU |
