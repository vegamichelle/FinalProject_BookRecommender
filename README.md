# 📚 Personalized Book Recommender — Production MLOps System

An end-to-end, production-grade MLOps system that recommends books from a
user's favourite titles. It covers the full lifecycle: experiment tracking and
a model registry (Weights & Biases), a FastAPI serving backend, a cloud
database for logging **and** a per-user recommendation cache (DynamoDB), a
user-facing Streamlit app, a **separate** model-monitoring dashboard, automated
testing + CI (GitHub Actions), and containerised deployment to AWS EC2.

> **Topic:** Personalized Book Recommender · **Dataset:** Amazon Review Data
> (Books subset). A synthetic sample with the same schema is bundled so the
> whole system runs with zero downloads.

---

## How the recommender works

Given a list of favourite titles, the model scores candidate books in three
tiers (it falls through only when the tier above produces nothing):

1. **Item-item collaborative filtering** — books that co-occur in many users'
   "liked" sets. Candidates are scored by summed cosine similarity to the
   user's favourites. This is the primary path.
2. **Content-based fallback** — TF-IDF over `title + author + genre`. Used for
   *cold-start* favourites that aren't in the ratings data.
3. **Popularity fallback** — most-liked books, when there's no usable signal.

The fitted model is a single picklable `Recommender` object
(`src/recommender.py`) containing the item-item neighbour lists, the TF-IDF
index, and catalogue metadata.

### Architecture

```
                        ┌─────────────────────────┐
                        │  Weights & Biases        │
                        │  Experiment tracking +   │
                        │  Model Registry          │
                        └───────────┬──────────────┘
                                    │ pull "production" model
                                    ▼
  ┌────────────┐  HTTP      ┌──────────────────┐   logs    ┌──────────────────┐
  │ Frontend   │───────────▶│ FastAPI Backend  │──────────▶│ DynamoDB          │
  │ (Streamlit)│ /recommend │ (EC2 instance A) │  cache    │ book-rec-logs     │
  │ EC2 inst. B│◀───────────│ /health /feedback│◀─────────▶│ book-rec-cache    │
  └────────────┘  books     │ /catalog         │           └────────┬─────────┘
                            └──────────────────┘                    │ read
                                                                    ▼
                                                 ┌───────────────────────────┐
                                                 │ Monitoring Dashboard      │
                                                 │ (Streamlit, EC2 inst. C)  │
                                                 │ reads DB directly — no API│
                                                 └───────────────────────────┘
```

The monitoring dashboard reads the logs table **directly** — never through the
backend API — as required.

## Repository layout

```
.
├── src/
│   ├── config.py          # all env-var configuration
│   ├── features.py        # pure preprocessing (title norm, matrix, metrics) — unit-tested
│   ├── recommender.py     # the Recommender model (fit / recommend), picklable artifact
│   ├── train.py           # training + evaluation + W&B tracking + registry
│   └── db.py              # DynamoDB logs + cache, with a local JSON fallback
├── backend/
│   ├── main.py            # FastAPI: /health /recommend /feedback /catalog /metrics
│   ├── model_loader.py    # load model from local file OR W&B registry
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/              # Streamlit user app (app.py, Dockerfile, requirements)
├── monitoring/            # Streamlit monitoring dashboard (reads DB)
├── data/generate_sample.py# synthetic books + ratings generator
├── tests/                 # test_features.py (unit) + test_api.py (integration)
├── .github/workflows/ci.yml
├── docker-compose.yml     # run all three services locally
├── requirements-dev.txt   # lint + train + test
├── ruff.toml
└── .env.example
```

---

## Quickstart (local, no AWS or W&B needed)

Defaults run offline: `DDB_LOCAL=1` uses JSON-backed stores and
`MODEL_SOURCE=local` loads a joblib file.

### Option 1 — bare Python

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
export PYTHONPATH=$PWD

# 1. Create data + train (writes artifacts/recommender.joblib)
python data/generate_sample.py --n-books 400 --n-users 3000
python -m src.train --no-wandb

# 2. Start the backend
uvicorn backend.main:app --reload --port 8000

# 3. Second shell: user app
BACKEND_URL=http://localhost:8000 streamlit run frontend/app.py --server.port 8501

# 4. Third shell: monitoring dashboard
streamlit run monitoring/dashboard.py --server.port 8502
```

Open http://localhost:8501 (app) and http://localhost:8502 (monitoring).

### Option 2 — docker compose

```bash
python data/generate_sample.py && python -m src.train --no-wandb   # once
docker compose up --build
```

---

## Phase 1 — Experiment tracking & model registry (W&B)

```bash
wandb login
export WANDB_PROJECT=mlops-book-recommender WANDB_ENTITY=<your-username>
python -m src.train --top-k 50 --min-interactions 3 --eval-k 10
```

Each run logs the **git commit**, **hyperparameters** (`sim_top_k`,
`min_interactions`, `eval_k`), **metrics** (`hit_rate@k`, `precision@k`,
catalogue `coverage`), and the **data version** (content hash + row count). The
fitted recommender is uploaded as an artifact and linked into the **Model
Registry** under `staging`.

**Promote to production** in the W&B UI (Model Registry → your model → add the
`production` alias). The backend serves whichever alias `WANDB_MODEL_ALIAS`
names.

> On the *synthetic* sample, expect `hit_rate@10 ≈ 0.35–0.40` (random is
> ≈ `k / catalogue_size`, ~0.025), which confirms real co-occurrence signal.
> Real Amazon data will differ — see below.

## Phase 2 — Backend + database

```bash
export MODEL_SOURCE=wandb WANDB_MODEL_ALIAS=production
export DDB_LOCAL=0 AWS_REGION=us-east-1
export DDB_LOGS_TABLE=book-rec-logs DDB_CACHE_TABLE=book-rec-cache
```

Every `/recommend` call is logged. When a `user_id` is supplied and
`CACHE_ENABLED=1`, repeat requests are served from the cache table instead of
being recomputed.

## Phase 3 — Frontend + monitoring

The dashboard reads the logs table and visualises recommendation latency over
time, recommended-genre distribution (target/popularity drift), request volume,
**cache-hit rate**, the recommendation source mix, and **live relevance**
(helpful-rate) from 👍/👎 feedback.

## Phase 4 — Testing & CI

```bash
ruff check .        # lint
pytest -q           # 16 unit + integration tests
```

`.github/workflows/ci.yml` runs ruff + pytest on every pull request to `main`.
**Make it blocking:** GitHub → Settings → Branches → add a rule for `main` →
*Require status checks to pass* → select `lint-and-test`. PRs then cannot merge
while checks fail.

---

## AWS deployment runbook

Three containers on **three separate EC2 instances**, plus two DynamoDB tables.

### 1. Create the DynamoDB tables

```bash
aws dynamodb create-table \
  --table-name book-rec-logs \
  --attribute-definitions AttributeName=request_id,AttributeType=S \
  --key-schema AttributeName=request_id,KeyType=HASH \
  --billing-mode PAY_PER_REQUEST --region us-east-1

aws dynamodb create-table \
  --table-name book-rec-cache \
  --attribute-definitions AttributeName=user_id,AttributeType=S \
  --key-schema AttributeName=user_id,KeyType=HASH \
  --billing-mode PAY_PER_REQUEST --region us-east-1
```

### 2. IAM role for the EC2 instances

Create a role (e.g. `book-rec-ec2-role`) allowing `dynamodb:PutItem`,
`dynamodb:UpdateItem`, `dynamodb:GetItem`, and `dynamodb:Scan` on both tables,
and attach it to the backend and monitoring instances. With the role attached,
no AWS keys need to live on the boxes.

### 3. Launch three EC2 instances

Amazon Linux 2023, `t3.small`. Security groups:

| Instance    | Inbound port | Source                     |
|-------------|--------------|----------------------------|
| backend     | 8000         | frontend + monitoring SGs  |
| frontend    | 8501         | your IP / 0.0.0.0          |
| monitoring  | 8502         | your IP / 0.0.0.0          |

Install Docker on each:

```bash
sudo yum update -y && sudo yum install -y docker git
sudo systemctl enable --now docker
sudo usermod -aG docker ec2-user   # re-login afterwards
```

### 4. Build & run each container

On **every** instance: `git clone <your-repo-url> && cd <repo>`.

**Backend instance:**
```bash
docker build -f backend/Dockerfile -t book-backend .
docker run -d --name backend -p 8000:8000 \
  -e MODEL_SOURCE=wandb -e WANDB_MODEL_ALIAS=production \
  -e WANDB_PROJECT=mlops-book-recommender -e WANDB_ENTITY=<you> -e WANDB_API_KEY=<key> \
  -e DDB_LOCAL=0 -e AWS_REGION=us-east-1 \
  -e DDB_LOGS_TABLE=book-rec-logs -e DDB_CACHE_TABLE=book-rec-cache \
  -e CACHE_ENABLED=1 \
  book-backend
```

**Frontend instance:**
```bash
docker build -f frontend/Dockerfile -t book-frontend .
docker run -d --name frontend -p 8501:8501 \
  -e BACKEND_URL=http://<backend-private-ip>:8000 book-frontend
```

**Monitoring instance:**
```bash
docker build -f monitoring/Dockerfile -t book-monitoring .
docker run -d --name monitoring -p 8502:8502 \
  -e DDB_LOCAL=0 -e AWS_REGION=us-east-1 -e DDB_LOGS_TABLE=book-rec-logs \
  book-monitoring
```

Visit `http://<frontend-public-ip>:8501` and
`http://<monitoring-public-ip>:8502`.

> **Serving without W&B at runtime:** train locally, then bake
> `artifacts/recommender.joblib` into the backend image (add
> `COPY artifacts ./artifacts`, set `MODEL_SOURCE=local`) or mount it as a
> volume.

---

## Using the real Amazon Books dataset

The bundled generator produces synthetic `books.csv` and `ratings.csv` with the
right shape. To train on real data, download the
[Amazon Review Data — Books subset](https://amazon-reviews-2023.github.io/)
and produce two CSVs:

* `books.csv` — `book_id,title,author,genre` (from the item metadata; use
  category as `genre`).
* `ratings.csv` — `user_id,book_id,rating` (from the review/rating records).

Then `python -m src.train --books path/books.csv --ratings path/ratings.csv`.
No model code changes are needed. On real, sparse data expect lower absolute
hit-rate and higher catalogue coverage sensitivity — tune `--min-interactions`
(raise it to drop long-tail noise) and `--top-k`.

---

## API reference

### `GET /health`  → `{ "status": "ok", "model_loaded": true }`

### `GET /catalog?n=30` → a few catalogue titles for the UI dropdown

### `POST /recommend`
```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{
        "favorite_titles": ["The Silent Forest", "The Crimson Machine"],
        "n": 5,
        "user_id": "user-123"
      }'
```
```json
{
  "request_id": "…",
  "recommendations": [
    {"book_id": 42, "title": "The Savage Cipher", "author": "Author C1",
     "genre": "Fantasy", "score": 0.31, "source": "collaborative"}
  ],
  "source": "collaborative",
  "cache_hit": false,
  "latency_ms": 2.1
}
```
Repeat the same call with the same `user_id` and `cache_hit` becomes `true`.

### `POST /feedback`
```bash
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{ "request_id": "<id from /recommend>", "helpful": true }'
```

Interactive docs at `http://<backend>:8000/docs`.

---

## Requirement checklist

| Spec requirement | Where |
|---|---|
| Experiment tracking + model registry | `src/train.py` (W&B) |
| FastAPI backend, `/recommend` + `/health` | `backend/main.py` |
| Cloud database, logs every request | `src/db.py` → `book-rec-logs` |
| Cache recommendations for frequent users | `src/db.py` → `book-rec-cache` |
| Frontend UI | `frontend/app.py` (Streamlit) |
| Monitoring dashboard on separate server, DB-only | `monitoring/dashboard.py` |
| Latency / target drift / feedback accuracy | dashboard charts |
| Unit + integration tests (pytest) | `tests/` |
| CI on PRs: lint + tests, blocking | `.github/workflows/ci.yml` |
| Containerisation | three `Dockerfile`s + `docker-compose.yml` |
| Deploy to separate EC2 instances | runbook above |
