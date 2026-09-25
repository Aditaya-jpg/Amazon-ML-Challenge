# Amazon ML Challenge 2026 — Submission Package

This folder contains the self-contained deliverables for the **Amazon ML Challenge 2026: Business Entity Resolution**.

---

## Directory Structure

```text
submission/
├── Documentation_template.md    # Completed official documentation write-up
├── README.md                    # Package overview and zip archive instructions
├── code/                        # Production codebase
│   ├── README.md                # Reproduction and execution guide
│   ├── requirements.txt         # Pinned python dependencies
│   └── src/
│       ├── __init__.py
│       ├── preprocess.py        # Unicode NFKD stripping, legal suffix & address normalization
│       ├── blocking.py          # Multi-index candidate blocker (rare tokens, 3-grams, location keys)
│       ├── features.py          # RapidFuzz C++ accelerated pairwise feature extractor (14 dims)
│       ├── model.py             # Precision-biased LightGBM classifier with macro F_0.5 calibration
│       ├── pipeline.py          # Training and validation driver
│       └── generate_submission.py # Streaming test inference generator (1.73M entities)
└── utils/
    └── validate_submission.py   # Official submission format validator
```

---

## Output Prediction Files

The model generates predictions directly into the `output/` directory:
- `output/matching_results.tsv` (primary scored leaderboard file: `source1_entity_id\tmatched_entity_ids`)
- `output/candidate_pairs.tsv` (blocking candidate pool: `source1_entity_id\tcandidate_entity_ids`)

Both files strictly adhere to:
- UTF-8 tab-separated (`.tsv`) formatting.
- Exactly 1,732,544 rows (one row per test Source 1 entity).
- Strict subset constraint (`matched_entity_ids ⊆ candidate_entity_ids`).

---

## Local Verification

Run the validator before submitting:
```bash
python utils/validate_submission.py \
    --matching ../output/matching_results.tsv \
    --candidate ../output/candidate_pairs.tsv \
    --test-dir ../student_resource/dataset/test
```

Expected output:
```text
PASS — no blocking issues found. Safe to submit.
```

---

## Packaging Deliverables for Portal Upload

To produce the official challenge zip file:
```bash
# Package into <TeamName>_Amazon_ML_Challenge_2026.zip
zip -r Aditaya_Amazon_ML_Challenge_2026.zip \
    Documentation_template.md \
    code/ \
    ../output/matching_results.tsv \
    ../output/candidate_pairs.tsv
```
