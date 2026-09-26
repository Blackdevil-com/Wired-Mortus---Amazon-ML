# CASCADE-ER v2: Stage 1 — WIDE BLOCKING

High-Recall Candidate Generation Engine for Large-Scale Business Entity Resolution.
Built with **Antigravity** and optimized for **Google Colab** execution.

---

## 1. Pipeline Architecture

```text
DATASET INPUTS
  train: train_source1.tsv, train_source2.tsv, train_source3.tsv, train_ground_truth.tsv
  test:  test_source1.tsv,  test_source2.tsv,  test_source3.tsv
                   │
                   ▼
STAGE 0: PREPROCESSING & NORMALIZATION (Stage 0)
                   │
                   ▼
STAGE 1: WIDE BLOCKING [THIS STAGE]
                   │
                   ▼
INTERNAL CANDIDATE UNION (`stage1_candidate_union.tsv`)
                   │
                   ▼
STAGE 2: SUPERVISED META-BLOCK (Stage 2 - Downstream)
                   │
                   ▼
STAGE 3: HEAVY FEATURE EXTRACTION (Stage 3 - Downstream)
                   │
                   ▼
STAGE 4: CLASSIFICATION & DECISION (Stage 4 - Downstream)
                   │
                   ▼
matching_results.tsv
```

---

## 2. Stage 1 Objective: MAXIMIZE CANDIDATE RECALL

Stage 1 discovers all plausible matches between Query Entities ($S_1$) and Candidate Pools ($S_2$ and $S_3$) by unioning six complementary blocking channels:

1. **Exact Normalized Name Block (`exact_name`)**: Exact match on `business_name_normalized` within the same country.
2. **Core Name Block (`core_name`)**: Strips corporate legal suffixes (`Inc`, `LLC`, `Ltd`, `Pvt`, `GmbH`, `SARL`, etc.) to find corporate variants (e.g., `ABC Private Limited` $\leftrightarrow$ `ABC Ltd`).
3. **House Number + Street Token Block (`house_street`)**: Matches identical house numbers with overlapping street tokens (e.g., `1795 Westchester Drive` $\leftrightarrow$ `1795 Westchester Dr`).
4. **Postal / Zip Code Block (`postal`)**: Matches identical postal codes / PIN codes within the same country.
5. **Character-level TF-IDF Retrieval (`tfidf`)**: Character n-grams (3–5) with Top-K sparse cosine retrieval ($K=30$).
6. **Multilingual Dense Embeddings / FAISS (`embedding`)**: Pretrained multilingual sentence transformers (`paraphrase-multilingual-MiniLM-L12-v2`) with Top-K FAISS retrieval ($K=20$).

---

## 3. Output Schema (`stage1_candidate_union.tsv`)

| Column | Type | Description |
|---|---|---|
| `s1_entity_id` | string | Query entity identifier from $S_1$ |
| `candidate_entity_id` | string | Candidate entity identifier from $S_2$ or $S_3$ |
| `candidate_source` | string | Source table (`S2` or `S3`) |
| `country` | string | Normalized country code |
| `blocking_channels` | JSON array | List of channels that found this pair (e.g. `["exact_name", "tfidf", "embedding"]`) |
| `blocking_channel_count` | int | Total number of channels discovering the pair |
| `tfidf_rank` | int / null | 1-indexed retrieval rank from TF-IDF |
| `tfidf_similarity` | float / null | TF-IDF cosine similarity score |
| `embedding_rank` | int / null | 1-indexed retrieval rank from dense embeddings |
| `embedding_similarity` | float / null | Embedding cosine similarity score |

---

## 4. Project Directory Structure

```text
entity_resolution_preprocessing/
│
├── data/
│   ├── raw/
│   │   ├── train/                 # Raw training files (train_source1.tsv, ..., train_ground_truth.tsv)
│   │   └── test/                  # Raw test files (test_source1.tsv, ...)
│   ├── processed/
│   │   ├── train/                 # Stage 0 preprocessed train TSVs
│   │   └── test/                  # Stage 0 preprocessed test TSVs
│   ├── candidates/                # Generated Stage 1 candidate union
│   │   └── stage1_candidate_union.tsv
│   └── cache/                     # Embedding checkpoints (.npy)
│
├── src/
│   ├── preprocessing/             # Stage 0 Normalization modules
│   │   ├── unicode_normalizer.py
│   │   ├── text_normalizer.py
│   │   ├── tokenizer.py
│   │   └── pipeline.py
│   │
│   ├── blocking/                  # Stage 1 Wide Blocking modules
│   │   ├── __init__.py
│   │   ├── config.py              # Centralized hyperparameters & channel toggles
│   │   ├── exact_name.py          # Channel 1: Exact Name Block
│   │   ├── core_name.py           # Channel 2: Core Name Block
│   │   ├── house_street.py        # Channel 3: House + Street Token Block
│   │   ├── postal.py              # Channel 4: Postal / Zip Code Block
│   │   ├── tfidf_blocking.py      # Channel 5: Character TF-IDF Retrieval
│   │   ├── embedding_blocking.py  # Channel 6: Dense Embedding Retrieval
│   │   ├── faiss_index.py         # FAISS vector index wrapper
│   │   ├── candidate_union.py     # Multi-channel union & deduplication
│   │   └── pipeline.py            # End-to-end blocking pipeline
│   │
│   └── evaluation/                # Ground-truth evaluation modules
│       ├── __init__.py
│       └── blocking_recall.py     # Overall union & per-channel recall scoring
│
├── tests/                         # 127 Unit & Integration tests
│   ├── test_exact_name.py
│   ├── test_core_name.py
│   ├── test_house_street.py
│   ├── test_postal.py
│   ├── test_tfidf_blocking.py
│   ├── test_embedding_blocking.py
│   ├── test_candidate_union.py
│   ├── test_blocking_pipeline.py
│   ├── test_blocking_recall.py
│   ├── test_text_normalizer.py
│   ├── test_tokenizer.py
│   └── test_unicode_normalizer.py
│
├── notebooks/
│   └── stage1_colab_runner.ipynb  # Automated Google Colab Execution Notebook
│
├── requirements-colab.txt
├── requirements-local.txt
├── run_preprocessing.py           # Stage 0 CLI entry point
└── run_stage1_blocking.py         # Stage 1 CLI entry point
```

---

## 5. Execution Instructions

### Local Execution

```bash
# 1. Install dependencies
pip install -r requirements-local.txt

# 2. Run unit & integration tests
pytest tests/ -v

# 3. Run Stage 1 Wide Blocking on Training data
python run_stage1_blocking.py \
    --split train \
    --s1 data/processed/train/train_source1_preprocessed.tsv \
    --s2 data/processed/train/train_source2_preprocessed.tsv \
    --s3 data/processed/train/train_source3_preprocessed.tsv \
    --ground-truth data/raw/train/train_ground_truth.tsv \
    --output data/candidates/stage1_candidate_union.tsv

# 4. Run Stage 1 Wide Blocking on Testing data
python run_stage1_blocking.py \
    --split test \
    --s1 data/processed/test/test_source1_preprocessed.tsv \
    --s2 data/processed/test/test_source2_preprocessed.tsv \
    --s3 data/processed/test/test_source3_preprocessed.tsv \
    --output data/candidates/stage1_candidate_union_test.tsv
```

### Google Colab Execution

1. Open `notebooks/stage1_colab_runner.ipynb` in Google Colab (enable GPU under Runtime $\rightarrow$ Change runtime type).
2. Execute the notebook top-to-bottom.
3. Intermediate embeddings are automatically cached to Google Drive so session reconnects do not repeat heavy computations.
