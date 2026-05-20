# 🏠 Real Estate Recommendation System — Datathon 2026 (Vòng Chung Kết)

> **Cuộc thi**: DATATHON - VÒNG CHUNG KẾT (Chợ Tốt)
> **Bài toán**: Dự đoán 10 bất động sản (item_id) mà mỗi user sẽ liên hệ (view_phone, contact_chat, contact_zalo, contact_sms) trong giai đoạn test.
> **Metric**: Recall@10
> **Best Score**: 0.0344 (v14)

---

## 📁 Cấu trúc Project

```
├── config/
│   └── settings.py                 # Pipeline configuration (paths, hyperparameters)
├── scripts/
│   ├── preprocess.py               # Step 1: Aggregate raw events → compact cache
│   ├── train.py                    # Step 2: Train ALS + SegPop models
│   ├── inference.py                # Step 3: Generate submission.csv
│   ├── evaluate.py                 # Offline evaluation (Recall@K, NDCG@K)
│   ├── run_gpu.sh                  # GPU launcher (sets LD_LIBRARY_PATH for CUDA)
│   ├── train_reranker.py           # Train LightGBM reranker (experimental)
│   └── generate_cold_prefs.py      # Generate cold user preferences from pageviews
├── src/
│   ├── core/
│   │   └── base.py                 # Abstract base classes
│   ├── data/
│   │   ├── loader.py               # FactUserEventsLoader (streaming parquet)
│   │   ├── preprocessor.py         # DataPreprocessor (cache builder)
│   │   └── loaders/
│   │       └── als_matrix_builder.py
│   ├── models/
│   │   ├── candidates/
│   │   │   ├── light_als.py        # ⭐ ALS collaborative filtering (implicit, GPU)
│   │   │   ├── segment_popularity.py # ⭐ SegPop cold-start fallback
│   │   │   ├── pageview_replay.py  # Replay recently viewed items
│   │   │   ├── cocontact.py        # Co-contact graph recommender
│   │   │   ├── intent_recommender.py # Intent-based (city+category matching)
│   │   │   ├── user_knn.py         # User-KNN co-occurrence CF
│   │   │   ├── seller_recommender.py # Seller expansion
│   │   │   ├── item2item.py        # Item co-occurrence (session-based)
│   │   │   ├── als_recommender.py  # Legacy ALS wrapper
│   │   │   ├── implicit_base.py    # Implicit base class
│   │   │   └── bpr_recommender.py  # BPR recommender (experimental)
│   │   ├── ensemble/
│   │   │   ├── cascade_generator.py # ⭐ Budget-based cascade candidate generator
│   │   │   └── ensemble_generator.py # Legacy ensemble
│   │   ├── rankers/
│   │   │   └── lgbm_ranker.py      # LightGBM LambdaRank reranker
│   │   ├── rerankers/
│   │   │   └── multi_objective.py  # Multi-objective reranker
│   │   └── baselines/
│   │       └── trending.py         # Burst trending recommender
│   ├── features/
│   │   ├── feature_engineer.py     # Feature engineering pipeline
│   │   ├── interaction_matrix.py   # Interaction matrix builder
│   │   ├── base.py                 # Feature extractor base
│   │   ├── feature_context.py      # Feature context
│   │   ├── cold_start.py           # Cold-start user profiler
│   │   └── extractors/             # Feature extractors for reranker
│   ├── evaluation/
│   │   ├── metrics.py              # Recall@K, NDCG@K, build_ground_truth
│   │   └── health_metrics.py       # Coverage, diversity, fairness metrics
│   ├── pipeline/
│   │   ├── training_pipeline.py    # End-to-end training orchestrator
│   │   ├── inference_pipeline.py   # Legacy inference pipeline
│   │   └── data_forensics.py       # Data quality checks
│   ├── utils/
│   │   ├── logging.py              # Logger setup
│   │   ├── profiler.py             # Data profiling
│   │   ├── plotting.py             # Visualization helpers
│   │   ├── polars_utils.py         # Polars utilities
│   │   └── report_writer.py        # Markdown report generator
│   └── eda/                        # EDA scripts & reports (22 rounds)
│       ├── .claude/                # AI agent state tracking
│       ├── round_01_*.py ... round_17_*.py
│       └── reports/                # Generated EDA reports
├── .gitignore
├── pyproject.toml
├── requirements.txt
├── uv.lock
└── README.md                       # ← This file
```

---

## 🚀 Quickstart — Tạo submission (Best Solution v14)

### Prerequisites

- Python 3.11+
- NVIDIA GPU (CUDA 13.x) — for ALS training
- 32GB+ RAM
- [uv](https://docs.astral.sh/uv/) package manager

### 1. Setup Environment

```bash
# Clone repo
git clone https://github.com/vtnguyen04/Real-Estate-Recomendation-System.git
cd Real-Estate-Recomendation-System

# Install dependencies
uv sync

# Verify GPU
.venv/bin/python -c "import implicit; print('implicit OK')"
```

### 2. Data Setup

Đặt data theo cấu trúc (cấu hình trong `config/settings.py`):

```
/home/db/rc/datathon/
├── train/
│   ├── fact_user_events/           # ~500 parquet files, 41GB
│   │   ├── part-00000.parquet
│   │   └── ...
│   ├── dim_listing/                # ~40 parquet files
│   │   ├── part-00000.parquet
│   │   └── ...
│   ├── fact_listing_snapshot/      # Daily snapshots
│   └── fact_post_contact_interactions/ # Contact aggregations
└── test/
    └── test_users.parquet          # 161,568 test user IDs
```

> ⚠️ Nếu data ở path khác, sửa `config/settings.py` → `train_path` và `test_path`.

### 3. Step 1 — Preprocessing (Cache Build)

```bash
# Aggregate 41GB raw events → compact cache files (~4GB total)
# Thời gian: ~2 phút
.venv/bin/python scripts/preprocess.py
```

Output tại `.cache/`:
| File | Size | Nội dung |
|------|------|----------|
| `contact_pairs.parquet` | 733MB | 13M user-item contact pairs (login-only) |
| `als_contact_pairs.parquet` | 723MB | ALS training pairs |
| `als_pageview_pairs.parquet` | 1.7GB | Pageview pairs (cho ViewALS, không dùng) |
| `session_items.parquet` | 1.1GB | Session co-occurrence |
| `cold_user_prefs.parquet` | 33MB | 902K cold user city/category preferences |
| `date_range.parquet` | <1KB | Min/max dates |

### 4. Step 2 — Training (ALS + SegPop)

```bash
# Train với GPU (~1 phút)
bash scripts/run_gpu.sh train
```

> ⚠️ **QUAN TRỌNG**: Training sẽ overwrite `segpop.pkl`. Phải restore recency version sau:
> ```bash
> # Nếu chưa có segpop_recency.pkl, bỏ qua bước này
> cp outputs/models/segpop_recency.pkl outputs/models/segpop.pkl
> ```

> ⚠️ **OOM WARNING**: ViewALS (step 4/8) sẽ tràn RAM trên máy 32GB.
> Kill process ngay khi thấy `[4/8] Fitting ViewALS` — ContactALS đã save xong.
> ```bash
> # Monitor và kill khi ViewALS bắt đầu:
> pkill -f "scripts/train.py"
> ```

Output tại `outputs/models/`:
| File | Size | Model |
|------|------|-------|
| `als/als.npz` | 1.5GB | ALS user/item factors (256 dims) |
| `als/als_matrix.npz` | 84MB | Sparse interaction matrix |
| `als/als_meta.pkl` | 262MB | User/item ID mappings |
| `segpop_trained.pkl` | 6MB | Alltime SegPop (KHÔNG dùng) |
| `segpop.pkl` | 4.4MB | ⭐ Recency SegPop (dùng cho inference) |

### 5. Step 3 — Inference (Generate Submission)

```bash
# Generate submission.csv (~2.5 phút)
bash scripts/run_gpu.sh inference
```

Output:
- `submission.csv` — 1,615,680 rows (161,568 users × 10 items)
- Format: `ID, user_id, rank, item_id`

### 6. Validate & Submit

```bash
# Validate format
.venv/bin/python -c "
import polars as pl
df = pl.read_csv('submission.csv')
assert len(df) == 1_615_680
assert list(df.columns) == ['ID', 'user_id', 'rank', 'item_id']
print('✅ Valid')
"

# Submit to Kaggle
uv run kaggle competitions submit \
  -c datathon-chung-ket \
  -f submission.csv \
  -m "v14: ALS 256f + recency SegPop + cascade"
```

---

## 🏗️ Solution Architecture (v14 — Best: 0.0344)

### Pipeline Overview

```
fact_user_events (41GB, 161M rows)
       │
       ▼ [preprocess.py]
  .cache/ (login-only contacts, pageviews, sessions)
       │
       ▼ [train.py]
  ContactALS (256 factors, 30 iterations, GPU)
  + SegPop (recency-weighted popularity by city+category)
       │
       ▼ [inference.py]
  Budget-based Sequential Cascade:
    ALS → Intent → PageviewReplay → UserKNN → CoContact
    → SellerExpansion → RecentCC → SegPop (fallback)
       │
       ▼
  submission.csv (161,568 users × 10 items)
```

### User Segmentation

| Segment | Count | % | Strategy |
|---------|-------|---|----------|
| **Warm** (có contact history) | 54,502 | 33.7% | ALS → Intent → cascade |
| **Cold** (có pageview prefs) | 3,651 | 2.3% | Intent → SegPop (city+cat) |
| **Blind** (zero data) | 103,415 | 64.0% | SegPop (hash-based segment) |

### Candidate Sources

| # | Source | Warm Users | Cold Users | Key Idea |
|---|--------|-----------|------------|----------|
| 1 | **ALS** | ~13K items | 0 | Collaborative filtering on login contacts |
| 2 | **IntentRecommender** | ~6K items | ~6K items | Match user's city+category intent |
| 3 | **PageviewReplay** | minimal | 0 | Recently viewed items |
| 4 | **UserKNN** | ~100 items | 0 | Co-occurrence similarity |
| 5 | **CoContact** | ~700 items | 0 | Graph-based co-contact |
| 6 | **SellerExpansion** | ~50 items | 0 | Other items from same seller |
| 7 | **RecentCC** | ~900 items | ~900 items | Recent popular by city+cat |
| 8 | **SegPop** | ~32K items | ~32K items | Recency-weighted popularity fallback |

### Key Design Decisions

1. **`is_login == "login"` filter**: MUST keep. Non-login events are noise (INS-057: removing = -59% score)
2. **Recency SegPop > Alltime SegPop**: 50.3% GT contacts on items ≤7 days old (INS-051)
3. **ALS budget=0 for als_view**: Pageview CF dilutes candidate pool (INS-047)
4. **Sequential cascade > round-robin**: Priority order matters (INS-046)

---

## 📊 Experiment History

| Version | Score | Key Changes | Status |
|---------|-------|-------------|--------|
| v5 | 0.0003 | Cascade V5 (glob bug) | ❌ |
| v6 | 0.0004 | Fix items, all users | ❌ |
| v10 | 0.0340 | ALS 256f + recency SegPop | ✅ |
| v11 | 0.0048 | + LightGBM reranker (wrong segpop bug) | ❌ |
| v12 | 0.0050 | + offset diversity (wrong segpop) | ❌ |
| v13 | 0.0140 | Remove is_login filter (noise dilution) | ❌ |
| **v14** | **0.0344** | Rollback + clean ALS retrain | ✅ BEST |

---

## 🧠 Key Insights (Top 10)

1. **85.5% GT items are NEW to user** — CF is secondary, content-based is critical
2. **91.9% GT items match user's city** — Location-first recommendation
3. **64% test users are completely blind** — Zero training data, cold-start is #1 challenge
4. **ALS density > size** — 16.1 contacts/user (login) > 7.5 (all users). Quality > quantity
5. **Recency > alltime popularity** — 50.3% contacts on items ≤7 days old
6. **SegPop city bug cost 96K users** — String mismatch between train/dim_listing
7. **Budget sequential union > round-robin** — Priority cascade +33% Recall@200
8. **als_view (pageview CF) = noise** — Disabling improves Recall@200 +5.4%
9. **Offline eval ≠ leaderboard** — Offline only measures warm users (33.7%)
10. **is_login filter = quality gate** — Non-login events destroy ALS embeddings

---

## ⚙️ Configuration

Key settings in `config/settings.py`:

```python
# ALS
als_factors = 256
als_iterations = 30
als_regularization = 0.01

# SegPop
segpop_cc_k = 500       # City+Category top items
segpop_segment_k = 500  # Per-segment top items
segpop_global_k = 500   # Global fallback

# Cascade budgets
n_cand_als = 100         # ALS candidates per user
n_cand_segpop = 200      # SegPop candidates per user
top_k = 10               # Final recommendations per user

# Validation
validation_days = 3      # Temporal split for offline eval
```

---

## 📋 Requirements

```
polars >= 1.0
implicit >= 0.7 (GPU support)
scipy
numpy
lightgbm (optional, for reranker)
pyarrow
```

Install: `uv sync` hoặc `pip install -r requirements.txt`

---

## 📝 License

Private — Datathon 2026 Competition
