# Entity Resolution — Stage 1: Preprocessing

A clean, modular Python preprocessing engine built for large-scale business entity resolution. Designed for local development in **Antigravity** and execution in **Google Colab**.

---

## 1. Project Architecture & Stage 1 Scope

The end-to-end entity-resolution architecture consists of four distinct stages:

```text
DATASET INPUTS
    │
    ├── train_source1.tsv
    ├── train_source2.tsv
    ├── train_source3.tsv
    ├── test_source1.tsv
    ├── test_source2.tsv
    └── test_source3.tsv
    │
    ▼
STAGE 1: PREPROCESSING            ← [THIS PROJECT]
    │
    ▼
STAGE 2: HYBRID BLOCKING          (Stage 2 - Not implemented here)
    │
    ▼
STAGE 3: FEATURE EXTRACTION       (Stage 3 - Not implemented here)
    │
    ▼
STAGE 4: CLASSIFICATION           (Stage 4 - Not implemented here)
    │
    ▼
matching_results.tsv
```

> **CRITICAL SCOPE BOUNDARY:**
> **Stage 1 is preprocessing only.** Candidate generation, hybrid blocking, TF-IDF, FAISS vector index, dense embeddings, Sentence Transformers, Levenshtein/Jaro-Winkler distances, cross-encoder scoring, classification (LightGBM/CatBoost), threshold optimization, and matching decisions belong to later stages and are **intentionally not implemented**.

---

## 2. Supported Datasets

The preprocessing engine processes six TSV sources:

```text
data/raw/train_source1.tsv  →  data/processed/train_source1_preprocessed.tsv
data/raw/train_source2.tsv  →  data/processed/train_source2_preprocessed.tsv
data/raw/train_source3.tsv  →  data/processed/train_source3_preprocessed.tsv
data/raw/test_source1.tsv   →  data/processed/test_source1_preprocessed.tsv
data/raw/test_source2.tsv   →  data/processed/test_source2_preprocessed.tsv
data/raw/test_source3.tsv   →  data/processed/test_source3_preprocessed.tsv
```

### Core Input Fields
* `entity_id` — Unique record identifier (e.g. `S1-925783039`)
* `business_name` — Raw business / entity name
* `business_address` — Raw postal address string
* `country` — Country code or country name

*Note: All original columns and original cell values are strictly preserved without mutation.*

---

## 3. Output Columns & Preprocessing Specifications

For each record, all original input fields are retained, and the following derived fields are generated:

| Column | Description | Example Input → Output |
|---|---|---|
| `business_name_nfkd` | NFKD-decomposed + lowercase name | `"Orelee's Barbershop"` → `"orelee's barbershop"` |
| `business_name_normalized` | Fully normalized name (punct → space) | `"Orelee's Barbershop"` → `"orelee s barbershop"` |
| `business_name_tokens` | Non-destructive token list as JSON | `["orelee", "s", "barbershop"]` |
| `business_address_nfkd` | NFKD-decomposed + lowercase address | `"1795 Westchester Dr, NC"` → `"1795 westchester dr, nc"` |
| `business_address_normalized` | Fully normalized address | `"1795 westchester dr nc"` |
| `business_address_tokens` | Non-destructive token list as JSON | `["1795", "westchester", "dr", "nc"]` |
| `country_normalized` | Normalized country code | `"US"` → `"us"` |
| `row_numbers` | All numeric sequences preserved from row | `["1795"]` |

---

## 4. Preprocessing Rules

### Text Normalization Pipeline (`normalize_text`)
1. **Handle Missing Values**: Coerce `None`, `NaN`, `math.nan` to `""`. Never produce `"nan"`, `"none"`, or `"null"`.
2. **Unicode NFKD Normalization**: Decompose composite characters (`unicodedata.normalize("NFKD", text)`).
3. **Strip Combining Marks (`Mn`)**: Strip Latin diacritics (`Café` → `cafe`) while preserving non-Latin scripts (Arabic, CJK, Devanagari).
4. **Lowercase Conversion**: Standard Unicode-aware lowercase.
5. **Punctuation & Separators to Whitespace**: Replace punctuation (`+`, `/`, `-`, `'`, `,`, `.`) with whitespace so tokens are never accidentally concatenated (`Orelee's` → `orelee s`, `B+ Retail` → `b retail`).
6. **Preserve Numbers & Alphanumerics**: Digits and mixed alphanumeric codes (`1795`, `2100`, `A12B`, `Unit7`, `B2`) are preserved without loss.
7. **Normalize Whitespace**: Collapse multiple whitespace runs and strip leading/trailing spaces.

---

## 5. Directory Structure

```text
entity-resolution/
│
├── data/
│   ├── raw/                       # Raw TSV dataset files
│   └── processed/                 # Generated Stage 1 preprocessed TSV files
│
├── src/
│   └── preprocessing/
│       ├── __init__.py            # Package exports
│       ├── unicode_normalizer.py  # Safe coercion & NFKD normalization
│       ├── text_normalizer.py     # Text normalization pipeline
│       ├── tokenizer.py           # Whitespace tokenizer & number extractor
│       └── pipeline.py            # Streaming chunk processor & validator
│
├── tests/
│   ├── test_unicode_normalizer.py # Unicode & safe_str unit tests
│   ├── test_text_normalizer.py    # Text normalizer unit tests
│   ├── test_tokenizer.py          # Tokenizer & number extraction tests
│   └── test_pipeline.py           # End-to-end pipeline integration tests
│
├── notebooks/
│   └── stage1_colab_runner.ipynb  # Google Colab execution notebook
│
├── requirements-colab.txt         # Lightweight Colab dependencies
├── requirements-local.txt         # Local development dependencies
├── README.md                      # Documentation
├── .gitignore                     # Git configuration
└── run_preprocessing.py           # CLI entry point
```

---

## 6. How to Run

### 6.1 Local Execution

1. **Install requirements:**
   ```bash
   pip install -r requirements-local.txt
   ```

2. **Run tests:**
   ```bash
   pytest tests/ -v
   ```

3. **Process all six datasets (Batch Mode):**
   ```bash
   python run_preprocessing.py
   ```

4. **Process a single file:**
   ```bash
   python run_preprocessing.py \
       --input data/raw/train_source1.tsv \
       --output data/processed/train_source1_preprocessed.tsv
   ```

---

### 6.2 Google Colab Execution

1. Open `notebooks/stage1_colab_runner.ipynb` in Google Colab.
2. Install dependencies:
   ```python
   !pip install -r requirements-colab.txt
   ```
3. Run tests and execute preprocessing:
   ```python
   !pytest tests/ -v
   !python run_preprocessing.py
   ```

---

### 6.3 Google Drive Support

To process datasets stored on Google Drive in Colab:
1. Mount Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
2. Place datasets at:
   `/content/drive/MyDrive/entity-resolution/data/raw/`
3. The Colab runner will read the datasets and write outputs to `data/processed/`.

---

## 7. What Stage 1 Intentionally Does NOT Do

Stage 1 strictly avoids downstream machine learning and matching logic:
* ❌ No stopword removal or stemming/lemmatization
* ❌ No translation, phonetic encoding (Soundex/Metaphone), or abbreviation expansion
* ❌ No candidate generation or blocking
* ❌ No TF-IDF, FAISS, or dense embeddings
* ❌ No Levenshtein or Jaro-Winkler distance calculations
* ❌ No classification models (LightGBM, CatBoost, Cross-Encoders)
* ❌ No matching decision files or candidate pairs
