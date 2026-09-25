# Business Entity Resolution Pipeline — Amazon ML Challenge 2026

## Overview
This package implements an end-to-end, reproducible Machine Learning pipeline for multi-source business entity resolution across millions of noisy commercial records (US, India, and France).

## Pipeline Architecture
1. **Normalization & Canonicalization (`src/preprocess.py`)**: Strips legal suffixes, normalizes diacritics/accents, cleans address abbreviations, extracts root brand names from URLs.
2. **Multi-Index Blocking (`src/blocking.py`)**: Partitions by country, uses inverted indexing on rare tokens, character 3-grams, and postal keys to retrieve high-recall candidates ($K \le 25$).
3. **Feature Engineering (`src/features.py`)**: Computes C++ accelerated string metrics (Levenshtein, Token Sort Ratio, Jaccard, number matching, domain matching).
4. **Scoring & Thresholding (`src/model.py`)**: Scores candidate pairs using LightGBM and applies an optimal precision-heavy decision threshold tuned for macro $F_{0.5}$.
5. **Output Generation & Validation (`src/pipeline.py`)**: Produces `matching_results.tsv` and `candidate_pairs.tsv` and verifies format with `validate_submission.py`.

## Quick Start
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run end-to-end training and inference
python src/pipeline.py --train-dir ../../student_resource/dataset/train --test-dir ../../student_resource/dataset/test --output-dir ../../output
```
