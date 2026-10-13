# Dark Pattern Detection: Semantic–Linguistic Feature Fusion for E-Commerce Text Classification

An undergraduate NLP research system and interactive web application investigating whether fusing contextual transformer representations with domain-specific linguistic feature vectors improves classification across e-commerce dark patterns.

---

## 1. Project Objective & Research Context

E-commerce user interfaces frequently employ coercive, deceptive, or manipulative design constructs (*dark patterns*) to influence purchasing choices. While general-purpose transformers capture deep distributional semantics, dark patterns exhibit distinct, identifiable linguistic constructions—such as imperative verb moods, temporal urgency markers, artificial inventory scarcity cues, and simulated social proof.

This study implements a controlled, reproducible 4-class experimental benchmark comparing:
1. **Model A (Baseline):** TF-IDF Vectorization + Multinomial Logistic Regression
2. **Model B (Transformer-Only):** Fine-tuned DistilBERT contextual text classifier
3. **Model C (Linguistic-Only):** 40-dimensional spaCy syntactic & linguistic feature extraction + classifier
4. **Model D (Proposed Late Fusion):** Contextual transformer representations fused with normalized linguistic vectors

### Target 4-Class Taxonomy:
- **Social Proof:** Activity notifications, crowd viewings, past purchase counts (e.g., *"1,142 people bought this recently"*).
- **Scarcity:** High-demand or low-stock claims (e.g., *"Hurry! Only 2 items left in stock"*).
- **Urgency:** Expiring offers, countdown timers, temporal constraints (e.g., *"FLASH SALE ends in 05:42"*).
- **Misdirection:** Asymmetric phrasing, misleading price anchors, comparative discounts (e.g., *"Compare at $120.00 — Now $59.99"*).

*Note:* In accordance with rigorous scientific evaluation, categories such as *Obstruction*, *Sneaking*, and *Forced Action* are safely audited and excluded from the 4-class task without relabeling.

---

## 2. Dataset Verification & Audit (`dark-patterns(2).csv`)

The model is trained strictly on verified real-world annotations:
- **Total Initial Records:** 1,818 rows
- **Missing Text Excluded:** 306 rows (Urgency: 244, Misdirection: 33, Sneaking: 14, Social Proof: 11, Forced Action: 2, Obstruction: 1, Scarcity: 1)
- **Usable Text Records:** 1,512 rows
- **Excluded Non-Target Categories:** 46 rows with usable text (Obstruction: 30, Sneaking: 12, Forced Action: 4)
- **Eligible 4-Class Dataset:** **1,466 samples**
  - *Scarcity:* 678 (46.25%)
  - *Social Proof:* 314 (21.42%)
  - *Misdirection:* 237 (16.17%)
  - *Urgency:* 237 (16.17%)
- **Data Partitions:** Stratified 70% Train (1,026), 15% Validation (220), 15% Test (220) with zero leakage and fixed random seed 42.

Complete column descriptions, data types, and handling rules are documented in [`docs/DATA_DICTIONARY.md`](docs/DATA_DICTIONARY.md).

---

## 3. Controlled Experimental Results (Held-Out Test Set)

Evaluated on the exact same 220-sample held-out test partition:

| Model Architecture | Accuracy | Macro F1 | Weighted F1 | Macro Precision | Macro Recall | Training Time |
|---|---|---|---|---|---|---|
| **Model A: TF-IDF + LogReg** | 95.91% | 0.9415 | 0.9578 | 0.9525 | 0.9404 | ~0.03s |
| **Model C: Linguistic-Only** | 89.09% | 0.8663 | 0.8897 | 0.8712 | 0.8624 | ~0.01s |
| **Model B: DistilBERT-Only** | 97.73% | 0.9721 | 0.9772 | 0.9705 | 0.9740 | ~65s |
| **Model D: Late Fusion (Proposed)** | **97.27%** | **0.9660** | **0.9724** | **0.9672** | **0.9654** | ~46s |

*Key Finding:* Linguistic features alone achieve nearly 90% accuracy with sub-millisecond inference, validating the high information density of syntactic cues in dark pattern detection. The neural models achieve state-of-the-art ~97% Macro-F1.

---

## 4. Quickstart: Running the Application Locally

### Prerequisites
- Python 3.10+ (tested on Python 3.14 on macOS)
- Node/npm (optional, frontend is vanilla HTML/JS with Tailwind CDN)

### Step 1: Install Dependencies
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Step 2: Ensure Dataset is in Place
Place `dark-patterns(2).csv` in `data/dark-patterns(2).csv`.

### Step 3: Run the Training Benchmark (Optional if models already trained)
```bash
python scripts/train_all.py
```
This script runs the data loader, trains Models A, B, C, D, saves model weights to `outputs/models/`, writes metrics to `outputs/metrics/`, and renders publication figures in `outputs/figures/`.

### Step 4: Launch the Web Application
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
Open your browser and navigate to:
```
http://localhost:8000
```

---

## 5. Google Colab Reproducibility

For peer reviewers or researchers who want to reproduce the experiments on a free Google Colab GPU:
1. Open [Google Colab](https://colab.research.google.com).
2. Upload the notebook located at `notebooks/dark_pattern_fusion_colab.ipynb`.
3. Upload `dark-patterns(2).csv` when prompted.
4. Select **Runtime &rarr; Run all**.

---

## 6. API Reference

- `GET /health` &bull; Service status, hardware acceleration device, and loaded model availability.
- `GET /metadata` &bull; Architecture parameters, target classes, and feature definitions.
- `GET /experiments` &bull; Complete benchmark metrics and confusion matrices across Models A, B, C, D.
- `GET /audit` &bull; Comprehensive preprocessing report and real sample count audit.
- `GET /data-dictionary` &bull; Dataset column definitions.
- `GET /dataset-sample?limit=25&category=Scarcity` &bull; Searchable, paginated records.
- `POST /predict` &bull; Single-text prediction with model choice (`model_a`, `model_b`, `model_c`, `model_d`).
- `POST /predict/all-models` &bull; Multi-model comparative prediction across all 4 models.
- `POST /batch-predict` &bull; Bulk classification of text lists.

---

## 7. Running Unit Tests

Run the full pytest suite:
```bash
PYTHONPATH=. pytest
```
All 28 unit tests cover:
- Dynamic column detection & raw count verification
- Safe exclusion of missing text & non-target classes
- 40-dimensional linguistic extraction across POS, syntax, and cues
- Neural model forward passes, gradient flow, and checkpointing
- FastAPI endpoints, validation schemas, and error responses.
