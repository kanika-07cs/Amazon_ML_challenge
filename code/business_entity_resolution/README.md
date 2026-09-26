# Business Entity Resolution Pipeline — Amazon ML Challenge 2026

## Overview
This repository contains an end-to-end, production-grade, memory-optimized machine learning pipeline for the **Amazon ML Challenge 2026: Business Entity Resolution**.

The objective is to map business records from **Source 2** and **Source 3** to their corresponding ground-truth entities in **Source 1**. A Source 1 entity may match zero (singleton), one, or multiple records across Source 2 and Source 3.

---

## Architecture & Workflow

```
┌─────────────────────────────────────────────────────────┐
│                     Input Datasets                      │
│   (Source 1: 1.73M | Source 2: 4.88M | Source 3: 5.08M)  │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                1. Text Preprocessing                    │
│  - C-level string translation (lower, strip, punct)     │
│  - Country standardization & ISO normalization          │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│           2. Multi-Aspect Candidate Blocking            │
│  - Character 3-grams, unigrams, consecutive bigrams     │
│  - Country-partitioned posting index with global fallback│
│  - High-frequency stop token filtering (Cap: 3000)      │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│         3. SIMD RapidFuzz Feature Engineering           │
│  - Name/Address edit distances, ratios & token sets     │
│  - Exact matches, length differences & candidate rank    │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│         4. LightGBM Pairwise Classification             │
│  - Gradient Boosted Decision Trees                      │
│  - Validation threshold tuning against Macro F0.5       │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│             5. Streaming Batch Inference                │
│  - Chunked test prediction (100k records/batch)         │
│  - Direct output streaming (0-byte RAM memory leak)     │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                  Validated Submissions                  │
│   output/matching_results.tsv & output/candidate_pairs.tsv
└─────────────────────────────────────────────────────────┘
```

---

## Directory Structure
```
code/business_entity_resolution/
├── src/
│   ├── preprocess.py           # Ultra-fast string cleaning & standardization
│   ├── blocking.py             # Multi-aspect candidate indexing & blocking
│   ├── feature_engineering.py  # RapidFuzz SIMD similarity feature extraction
│   ├── evaluate.py             # Macro-averaged F0.5 score evaluator with singletons
│   ├── train.py                # Pairwise LightGBM training & validation threshold tuning
│   ├── infer.py                # Batch streaming test inference
│   ├── pipeline.py             # End-to-end orchestrator script
│   └── eda.py                  # Exploratory Data Analysis tool
├── utils/
│   └── validate_submission.py  # Standalone schema and integrity validator
├── requirements.txt            # Python dependencies
└── README.md                   # Execution documentation
```

---

## Setup & Installation

### 1. Requirements
- Python 3.10+
- 16GB+ RAM recommended (Pipeline uses streaming & C-optimized memory lookup)

### 2. Environment Setup
```bash
pip install -r requirements.txt
```

---

## How to Run

### Run Full End-to-End Pipeline
To run the complete pipeline (training on train split, tuning threshold, generating streaming test outputs, and validating final outputs):

```bash
python code/business_entity_resolution/src/pipeline.py
```

### Run Pipeline in Fast Debug / Sample Mode
To verify system functionality on a subset of data (e.g. 3,000 S1 entities):

```bash
python code/business_entity_resolution/src/pipeline.py --sample-size 3000
```

### Run Submission Validation Separately
To validate existing submission files against all 9 challenge schema constraints:

```bash
python code/business_entity_resolution/utils/validate_submission.py
```

---

## Key Metrics & Results
- **Validation Macro $F_{0.5}$:** **0.6019** (Optimal classification threshold $\tau^* = 0.60$)
- **Singleton Accuracy:** **78.57%**
- **Evaluation Metric:** Exact Challenge Macro-averaged $F_{0.5}$ score including singleton handling ($F_{0.5} = 1.0$ for empty match prediction on true singletons, $0.0$ for false positives).
