# cv4 — Plant Root Segmentation and Landmark Detection

[![CI](https://github.com/AhmedAlezabi1/root-segmentation-cv4/actions/workflows/ci.yml/badge.svg)](https://github.com/AhmedAlezabi1/root-segmentation-cv4/actions/workflows/ci.yml)

Computer-vision system for automated plant phenotyping at **NPEC** (Netherlands Plant Eco-phenotyping Centre). Given a single grayscale petri-dish image holding five seedlings, `cv4` segments the roots, finds each plant's main root tip, and reports the tip position with a confidence score. It ships as a Python package, a command-line tool, and an authenticated REST API with a Next.js dashboard, and it runs either fully locally or against a managed model endpoint in Azure ML.

![cv4 dashboard: detected root tips per lane with confidence scores](frontend/app/screenshots/inference_result.png)

## Project overview

This is the **production version** of a research pipeline I first built on my own. The research prototype (U-Net segmentation plus robot control) is in [plant-phenotyping-robotics](https://github.com/AhmedAlezabi1/plant-phenotyping-robotics). Here, a team of five turned it into a deployable system:

- **Model:** Residual Attention U-Net with patch-based inference and Hann-window blending (validation F1 ≈ 0.88)
- **Software engineering:** installable Python package, CLI, FastAPI REST API with JWT authentication, pytest suite, pre-commit hooks, CI
- **Product:** Next.js dashboard with a **human-in-the-loop feedback** loop to correct predictions
- **MLOps:** Docker, PostgreSQL + Alembic migrations, MLflow tracking, Azure ML training pipelines and managed endpoints, data-drift checks, monitoring and alerting

**Tech stack:** Python · TensorFlow/Keras · OpenCV · scikit-image · NetworkX · FastAPI · PostgreSQL · SQLAlchemy/Alembic · Next.js/TypeScript · Docker · Azure ML · Application Insights · MLflow · GitHub Actions · Sphinx

### My contributions

- **Model deployment on Azure ML:** wrote the scoring script and deployed the model as a **Kubernetes online endpoint with GPU inference**, added a **green deployment for versioned traffic splitting** (blue-green releases), autoscaling, and endpoint latency/performance measurement
- **Monitoring:** added **Application Insights** telemetry to both the backend and the frontend (incl. page-view tracking by route), wired it into the on-premise Portainer deployment, and wrote the [alert runbooks](docs/alert_runbooks.md)
- **Documentation & CI:** set up the **Sphinx documentation build and deployment to GitHub Pages**, wrote the project task documentation, and contributed architecture diagrams and README sections
- CLI training command fixes and inference improvements

*This is a cleaned copy of the team's original repository, published with my teammates' permission. Deployment workflows that depended on the original cloud resources are kept in [`.github/workflows-archive/`](.github/workflows-archive/) for reference.*

---

## Table of contents

- [What it does](#what-it-does)
- [How it works](#how-it-works)
- [Project structure](#project-structure)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Usage](#usage)
  - [Python API](#python-api)
  - [Command-line interface](#command-line-interface)
  - [REST API](#rest-api)
- [Configuration](#configuration)
- [Running the full stack with Docker](#running-the-full-stack-with-docker)
- [Database and migrations](#database-and-migrations)
- [Training and Azure ML pipelines](#training-and-azure-ml-pipelines)
- [Monitoring and drift detection](#monitoring-and-drift-detection)
- [Testing](#testing)
- [Development](#development)
- [Troubleshooting](#troubleshooting)
- [FAQ](#faq)
- [Further documentation](#further-documentation)
- [Authors](#authors)

---

## What it does

For each petri-dish image, the system returns:

- **Per-lane root tip coordinates**, in pixels.
- **A confidence score** for every detected tip, taken from the model's probability map at the tip pixel.
- **Segmentation masks** for the roots, both as a single binary mask and split into one mask per lane.

Each plate holds five plants arranged in vertical lanes numbered `0`–`4` from left to right. Every result contains exactly five entries; lanes with no detected root are returned explicitly with `has_root=false` rather than being omitted, so consumers can iterate over a fixed-size structure.

## How it works

The pipeline runs five stages in memory, passing NumPy arrays between them without writing intermediate files:

```
preprocessing → inference → postprocessing → segmentation → localization
```

1. **Preprocessing** loads the image, detects the petri dish, and crops and pads it to a consistent frame.
2. **Inference** runs a **Residual Attention U-Net** over 256 px patches (128 px stride, 50% overlap) with smooth Hann-window blending, and reconstructs a full-resolution probability map.
3. **Postprocessing** turns the probability map into a clean binary mask using hysteresis thresholding, morphological closing, and skeleton gap-bridging.
4. **Segmentation** splits the mask into per-lane plant instances using lane edges and a scoring heuristic that favours tall, central, downward-growing components.
5. **Localization** skeletonises each plant and runs **Dijkstra pathfinding** over the skeleton to trace the main root from the seedling to its tip, with vertical-bias penalties and separate strategies for very large and very small root systems.

The model and its tuned parameters were derived from the project's reference pipeline (`Task_8.ipynb`, validation F1 ≈ 0.88). All tunable values live in [`src/cv4/config.py`](src/cv4/config.py); see [`docs/pipeline_selection.md`](docs/pipeline_selection.md) for the selection rationale.

The inference stage runs in one of two modes (see [Configuration](#configuration)):

- **`local`** — the U-Net is loaded into the API process and runs in-process.
- **`remote`** — inference is delegated to an Azure ML Managed Endpoint, while all other stages still run locally.

## Project structure

```
src/cv4/
├── config.py             # Central configuration — every tunable parameter
├── preprocessing.py      # Load image, detect petri dish, crop, pad
├── inference.py          # Load U-Net, run patch-based prediction (local mode)
├── postprocessing.py     # Probability map → clean binary mask
├── segmentation.py       # Split mask into 5 per-lane instances
├── localization.py       # Main-root + tip detection via Dijkstra pathfinding
├── pipeline.py           # End-to-end orchestrator (run_pipeline)
├── cli.py                # CLI entry points: cv4-predict, cv4-train
├── db.py                 # SQLAlchemy engine + session factory
├── api/
│   ├── app.py            # FastAPI app, lifecycle, /predict, /health, /feedback, /auth
│   ├── auth.py           # JWT auth, PBKDF2 password hashing, current-user dependency
│   ├── dashboard.py      # /dashboard analytics routes
│   ├── endpoint.py       # Azure ML endpoint client (remote inference mode)
│   ├── models.py         # SQLAlchemy ORM models
│   └── schemas.py        # Pydantic request/response schemas
├── deployment/
│   └── score.py          # Azure ML scoring script (managed-endpoint entry point)
├── monitoring/
│   ├── baseline.py       # Confidence-baseline computation
│   ├── drift.py          # Drift detection
│   └── telemetry.py      # OpenTelemetry → Azure Monitor
└── training/
    ├── train.py          # Training entry point
    ├── model.py          # Residual Attention U-Net architecture
    ├── preprocess.py     # Raw images → 256 px training patches
    ├── data.py           # Dataset loading
    ├── evaluate.py       # Evaluation
    ├── losses.py         # Loss functions
    ├── metrics.py        # F1 and related metrics
    ├── register.py       # Model registration
    ├── pipeline.py       # Azure ML training-pipeline definition
    ├── config.py         # Training configuration
    └── mlflow_setup.py   # MLflow tracking setup

alembic/                  # Database migrations (6 versions)
components/               # Azure ML pipeline component specs (YAML)
configs/                  # Conda environment specs for Azure (cpu / training / inference)
docker/                   # Dockerfiles and docker-compose stack
docs/                     # Sphinx docs, architecture diagrams, runbooks
frontend/                 # Next.js web interface
scripts/                  # Azure ML and operational scripts
tests/                    # pytest suite (unit + integration)
weights/                  # Model weights (gitignored; downloaded separately)
```

Data flows left-to-right through the pipeline modules; the API and CLI are thin layers over `run_pipeline`, holding no algorithm logic of their own.

## Installation

### Prerequisites

- **Python 3.12** (the project pins `>=3.12,<3.13`).
- **[uv](https://docs.astral.sh/uv/)** for dependency management.
- Optional: **Docker** and **Docker Compose** to run the full stack (API + frontend + database).

### Clone and install

```bash
git clone https://github.com/AhmedAlezabi1/root-segmentation-cv4.git
cd 2025-26d-fai2-adsai-group-computervision4

uv sync              # create the environment and install locked dependencies
uv pip install -e .  # install cv4 in editable mode
```

Editable mode means code changes take effect without reinstalling. To install only the optional Azure tooling, add the group:

```bash
uv sync --group azure   # azure-ai-ml, azure-identity, azure-mgmt-compute
uv sync --group dev     # pre-commit, pytest, ruff, sphinx (for contributors)
```

### Model weights

The trained model and the lane-edge file are too large for git and are not in the repository. Download both from the team's shared OneDrive folder and place them in `weights/` at the repository root:

| File | Description | Size |
|---|---|---|
| `unet_256px.h5` | Trained Residual Attention U-Net weights | ~100 MB |
| `lane_edges.npy` | Precomputed lane-boundary positions | small |

If `lane_edges.npy` is not available, generate it:

```bash
uv run python scripts/generate_lane_edges.py
```

If a required file is missing at run time, the pipeline raises a clear error naming the file and the path it expects. To keep weights somewhere other than `weights/` (for example inside a container image), set `CV4_MODEL_DIR` — see [Configuration](#configuration).

### Verify the install

```bash
uv run cv4-predict --help
uv run pytest tests/ -q
```

## Quick start

**Python:**

```python
from cv4 import pipeline

result = pipeline.run_pipeline("path/to/dish.png")

print(f"Processed in {result['processing_time_seconds']:.2f}s")
for plant in result["plants"]:
    if plant["has_root"]:
        tip = plant["root_tip_mm"]
        print(f"Lane {plant['lane']}: tip at ({tip['x']:.1f}, {tip['y']:.1f}) mm, "
              f"confidence {plant['confidence']:.2f}")
    else:
        print(f"Lane {plant['lane']}: no root detected")
```

**Command line:**

```bash
uv run cv4-predict --image path/to/dish.png --visualize
```

## Usage

### Python API

The entry point is `cv4.pipeline.run_pipeline()`:

```python
def run_pipeline(
    image_path: str,
    model_path: str | None = None,      # defaults to config.DEFAULT_MODEL_PATH
    lane_edges: np.ndarray | None = None,
    lane_edges_path: str | None = None,  # defaults to config.DEFAULT_LANE_EDGES_PATH
    model=None,                          # pass a preloaded model to skip loading
) -> dict
```

#### Result structure

```python
{
    "image_path": str,                  # echoed input path
    "binary_mask": np.ndarray,          # full clean mask (unpadded)
    "plant_masks": list[np.ndarray],    # 5 per-lane masks
    "plants": [
        {
            "lane": int,                                   # 0–4
            "has_root": bool,
            "root_tip_px": {"x": int, "y": int} | None,    # None if no root
            "root_tip_mm": {"x": float, "y": float} | None,# None if no root
            "confidence": float | None,                    # None if no root
            "main_root_path": list[tuple[int, int]],       # [] if no root
        },
        # ... 5 entries, one per lane
    ],
    "processing_time_seconds": float,
    "plate_bbox": (x, y, side),         # crop position in the original image
}
```

When `has_root` is `False`, the coordinate and confidence fields are explicitly `None`, not zero. Always check `has_root` before reading them. Root-tip pixel coordinates are relative to the cropped plate; add `plate_bbox` x/y to map them back to the original image.

#### Reusing the model across calls

Loading the model costs roughly 30 seconds. In a long-running service, load it once and pass it in:

```python
import numpy as np
from cv4 import config, inference, pipeline

# Load once at startup
model = inference.load_model(config.DEFAULT_MODEL_PATH)
lane_edges = np.load(config.DEFAULT_LANE_EDGES_PATH)

# Reuse for every request
result = pipeline.run_pipeline("image.png", model=model, lane_edges=lane_edges)
```

### Command-line interface

The package installs two commands, `cv4-predict` and `cv4-train`.

#### `cv4-predict`

Runs the pipeline on one image, prints results, and writes a JSON file. Results are always saved (default location is `outputs/<image-name>_results.json`) and echoed to the terminal.

```bash
uv run cv4-predict --image path/to/dish.png
uv run cv4-predict --image dish.png --output results/run01.json
uv run cv4-predict --image dish.png --visualize
uv run cv4-predict --image dish.png --model weights/custom.h5 --output out.json --visualize
```

| Option | Required | Default | Description |
|---|---|---|---|
| `--image` | Yes | — | Path to the input image (PNG, JPG, or TIFF). |
| `--model` | No | `config` value | Path to model weights (`.h5`). |
| `--output` | No | `outputs/<name>_results.json` | Where to write the JSON results. |
| `--visualize` | No | off | Also save an overlay image with the mask (green) and tips (red dots, lane labels `L0`–`L4`). |

A run logs to `outputs/cv4.log` and ends with a summary such as `Done! 4/5 roots detected in 4.21s`. The `outputs/` folder is gitignored.

#### `cv4-train`

Trains the Residual Attention U-Net on paired image/mask data, logs to MLflow, and saves the model. The same `cv4.training.train` code runs both this command and the cloud pipeline, so behaviour is identical — only the compute and data source differ. Use `--smoke-test` for a fast end-to-end check; run full training on a GPU.

```bash
# Fast smoke test — caps at 2 steps per epoch
uv run cv4-train --data path/to/dataset --smoke-test

# Full run
uv run cv4-train --data dataset/ --epochs 50 --batch-size 32

# Custom run with MLflow tracking
uv run cv4-train --data dataset/ --augmentation strong --experiment-name roots --run-name exp_1
```

| Option | Default | Description |
|---|---|---|
| `--data` | required | Dataset root containing `train_images/`, `train_masks/`, `val_images/`, `val_masks/`. |
| `--model-output` | `outputs/trained_model.h5` | Where to save the trained model. |
| `--epochs` | `50` | Maximum training epochs (early stopping may end sooner). |
| `--batch-size` | `32` | Patches per training step. |
| `--learning-rate` | `1e-4` | Initial optimizer learning rate. |
| `--augmentation` | `mild` | Preset: `none`, `mild`, or `strong`. |
| `--experiment-name` | — | MLflow experiment name. |
| `--run-name` | — | MLflow run name. |
| `--smoke-test` | off | Quick validation run (2 steps/epoch, 2 val steps). |

The model trains on 256×256 patches, not whole plates — raw images are too large and mostly background. Preprocess raw images into patches first; see [Training and Azure ML pipelines](#training-and-azure-ml-pipelines).

### REST API

The HTTP API is a thin layer over `run_pipeline`. It handles authentication, validation, serialization, and feedback storage; all algorithm logic stays in the pipeline.

#### Starting the server

```bash
# Development (auto-reload)
uv run uvicorn cv4.api.app:app --reload

# Production (env vars injected by the platform)
uvicorn cv4.api.app:app --host 0.0.0.0 --port 8000 --workers 4
```

The server listens on `http://localhost:8000`. Interactive documentation is generated automatically at `/docs` (Swagger UI) and `/redoc` (ReDoc).

#### Authentication

The API uses **JWT authentication with httpOnly cookies**, backed by user accounts stored in the database. Passwords are hashed with PBKDF2-HMAC-SHA256 (600,000 iterations). A successful login sets `access_token` and `refresh_token` cookies; the response body carries only non-sensitive user info. Protected endpoints read the `access_token` cookie automatically — there is no `Authorization` header or API key to manage on the client.

| Method & path | Auth | Body | Purpose |
|---|---|---|---|
| `POST /auth/register` | No | `{email, password, password_confirm}` | Create an account (password 8–128 chars). |
| `POST /auth/login` | No | `{email, password}` | Log in; sets auth cookies. |
| `POST /auth/refresh` | Refresh cookie | — | Issue a new access token. |
| `POST /auth/logout` | Cookie | — | Clear auth cookies. |
| `GET /auth/me` | Cookie | — | Return the current user `{id, email, is_active}`. |
| `POST /auth/reset-password/request` | No | `{email}` | Begin a password reset. |
| `POST /auth/reset-password/confirm` | No | `{token, new_password, new_password_confirm}` | Complete a reset. |

In development (`CV4_ENV=development`), `POST /auth/reset-password/request` returns the reset token directly in the response so you can test without an email server. In production it only confirms that a message was sent.

#### Inference

**`POST /predict`** — requires authentication. Send the image as `multipart/form-data` with field name `image` (PNG, TIFF, or JPEG). Requests with a missing or non-image content type are rejected with `400` before any disk I/O.

```bash
# 1. Register (once)
curl -X POST http://localhost:8000/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email":"you@example.com","password":"your-password","password_confirm":"your-password"}'

# 2. Log in and store the cookies
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"you@example.com","password":"your-password"}' \
  -c cookies.txt

# 3. Predict using the stored cookies
curl -X POST http://localhost:8000/predict -b cookies.txt -F "image=@plate_scan.png"
```

The same flow in Python, using a session so cookies persist automatically:

```python
import requests

s = requests.Session()
s.post("http://localhost:8000/auth/login",
       json={"email": "you@example.com", "password": "your-password"})

with open("plate_scan.png", "rb") as f:
    resp = s.post("http://localhost:8000/predict", files={"image": f})

data = resp.json()
print(f"Prediction {data['prediction_id']} in {data['processing_time_s']}s")
for root in data["roots"]:
    if root["has_root"]:
        print(f"Lane {root['lane']}: tip {root['root_tip_px']} (confidence {root['confidence']:.2f})")
```

Response (`200 OK`):

```json
{
  "prediction_id": 42,
  "roots": [
    { "lane": 0, "has_root": true,  "root_tip_px": { "x": 145, "y": 890 }, "confidence": 0.9123 },
    { "lane": 1, "has_root": false, "root_tip_px": null, "confidence": null },
    { "lane": 2, "has_root": false, "root_tip_px": null, "confidence": null },
    { "lane": 3, "has_root": false, "root_tip_px": null, "confidence": null },
    { "lane": 4, "has_root": false, "root_tip_px": null, "confidence": null }
  ],
  "processing_time_s": 4.21,
  "crop": { "x": 312, "y": 480, "side": 1500 }
}
```

`root_tip_px` coordinates are in cropped-plate space. Add `crop.x` / `crop.y` to convert them to original-image pixels. Keep the `prediction_id` if you intend to submit feedback.

**`GET /health`** — no authentication. Returns `200` with `{ "status": "ok", "model_loaded": true, "inference_mode": "local" }` when ready, or `503` when the backend is not ready (model failed to load in local mode, or the endpoint is unconfigured in remote mode). In remote mode `model_loaded` is `false` because no local model is loaded. Orchestrators and load balancers use this endpoint to gate traffic.

#### Feedback (human-in-the-loop)

Feedback closes the loop for retraining: users confirm correct tips or supply corrected coordinates, which feed drift monitoring and future training rounds.

| Method & path | Body | Purpose |
|---|---|---|
| `POST /feedback` | `{prediction_id, lanes: [...]}` | Submit per-lane corrections or confirmations. Returns `201` with the assigned feedback IDs. |
| `POST /feedback/{prediction_id}/image` | image upload | Attach the source image to a prediction's feedback. |
| `GET /feedback/export` | — | Export collected feedback. |

Each lane entry is `{lane, is_correct, predicted_tip_x?, predicted_tip_y?, correct_tip_x?, correct_tip_y?, comment?}`. When `is_correct` is `false`, both `correct_tip_x` and `correct_tip_y` are required.

```bash
curl -X POST http://localhost:8000/feedback -b cookies.txt \
  -H "Content-Type: application/json" \
  -d '{
        "prediction_id": 42,
        "lanes": [
          {"lane": 0, "is_correct": false, "predicted_tip_x": 145, "predicted_tip_y": 890,
           "correct_tip_x": 148, "correct_tip_y": 895, "comment": "tip slightly lower"},
          {"lane": 1, "is_correct": true, "predicted_tip_x": 300, "predicted_tip_y": 750}
        ]
      }'
```

#### Dashboard

The `/dashboard` routes back the monitoring UI and return aggregated analytics:

| Path | Returns |
|---|---|
| `GET /dashboard/summary` | High-level KPI summary. |
| `GET /dashboard/activity` | Images processed per day (last 30 days). |
| `GET /dashboard/confidence` | Average confidence trend (last 30 days). |
| `GET /dashboard/correction-rate` | Correction rate per day (last 30 days). |
| `GET /dashboard/drift-status` | Current model-drift indicator. |

#### Error responses

All errors share one envelope: `{ "detail": "Human-readable description." }`.

| Code | Cause |
|---|---|
| `400 Bad Request` | Missing/invalid content type, undecodable image, or no petri dish detected. |
| `401 Unauthorized` | Missing or invalid authentication. |
| `500 Internal Server Error` | Unexpected error during inference — check server logs. |
| `502 Bad Gateway` | Remote endpoint returned an error (remote mode only). |
| `503 Service Unavailable` | Inference backend not ready — check `/health`. |
| `504 Gateway Timeout` | Remote endpoint did not respond in time (remote mode only). |
| `507 Insufficient Storage` | Not enough GPU/CPU memory to run inference (local mode only). |

## Configuration

Configuration comes from environment variables, read at startup. A `.env` file in the project root (or `docker/.env` for the stack) is loaded automatically via `python-dotenv`. The `.env` file is gitignored and must never be committed; copy `docker/.env.example` as a starting point.

**Core and model:**

| Variable | Default | Description |
|---|---|---|
| `CV4_ENV` | unset | Set to `development` to enable dev conveniences (dev JWT secret fallback, reset token in responses). |
| `CV4_MODEL_DIR` | `<repo>/weights` | Directory holding `unet_256px.h5` and `lane_edges.npy`. |
| `CV4_MODEL_VERSION` | unset | Optional model version label recorded with predictions. |
| `CV4_INFERENCE_MODE` | `local` | `local` runs the U-Net in-process; `remote` delegates to an Azure ML endpoint. |

**Authentication:**

| Variable | Default | Description |
|---|---|---|
| `JWT_SECRET_KEY` | — | Signing secret. **Required** unless `CV4_ENV=development`; the server refuses to start otherwise. Generate with `python -c "import secrets; print(secrets.token_hex(32))"`. |
| `JWT_ALGORITHM` | `HS256` | JWT signing algorithm. |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | `30` | Access-token lifetime. |
| `REFRESH_TOKEN_EXPIRE_DAYS` | `7` | Refresh-token lifetime. |
| `RESET_TOKEN_EXPIRE_MINUTES` | — | Password-reset token lifetime. |

**Database:**

| Variable | Description |
|---|---|
| `DATABASE_URL` | SQLAlchemy URL. SQLite works for local development; PostgreSQL via psycopg in production, e.g. `postgresql+psycopg://cv4:changeme@localhost:5433/cv4`. The driver is selected from the URL scheme. |

**Remote inference (Azure ML endpoint, `CV4_INFERENCE_MODE=remote`):**

| Variable | Description |
|---|---|
| `AZURE_ML_ENDPOINT_URL` | Managed-endpoint scoring URL. |
| `AZURE_ML_ENDPOINT_KEY` | Endpoint authentication key. |
| `AZURE_ML_DEPLOYMENT_NAME` | Target deployment name. |
| `CV4_ENDPOINT_TIMEOUT_S` | Per-request timeout before returning `504`. |

**Feedback, dashboard, and tracking:**

| Variable | Description |
|---|---|
| `CV4_FEEDBACK_IMAGE_DIR` | Directory where feedback images are stored. |
| `DASHBOARD_CACHE_TTL_SECONDS` | Cache lifetime for dashboard analytics. |
| `CV4_MLFLOW_TRACKING_URI` | MLflow tracking server URI. |
| `CV4_MLFLOW_EXPERIMENT` | Default MLflow experiment name. |

## Running the full stack with Docker

`docker/docker-compose.yml` brings up the API, the frontend, the database, and a one-shot migration step.

```bash
cd docker
cp .env.example .env          # set POSTGRES_PASSWORD at minimum
docker compose up --build
```

| Service | Port (host) | Description |
|---|---|---|
| `frontend` | `3000` | Next.js web interface. |
| `backend` | `8000` | FastAPI service. |
| `db` | `5433` → 5432 | PostgreSQL 16 (Alpine). Host port `5433` avoids clashing with a local Postgres. |
| `db-migrate` | — | Runs Alembic migrations on startup, then exits. |

Mount the model weights into the backend so it can load them, and point `CV4_MODEL_DIR` at the mount. Once up, the UI is at `http://localhost:3000`, the API at `http://localhost:8000`, and Swagger at `http://localhost:8000/docs`. For on-premise Portainer deployment, see the deployment guide in [`docs/`](docs/).

## Database and migrations

The schema is managed with **Alembic** (`alembic/versions/`, six migrations covering predictions, users, prediction logs, confidence baselines, drift events, and feedback images). With `DATABASE_URL` set:

```bash
uv run alembic upgrade head            # apply all migrations
uv run alembic revision --autogenerate -m "describe change"   # create a new one
uv run alembic downgrade -1            # roll back one step
```

In the Docker stack, `db-migrate` runs `alembic upgrade head` automatically before the backend starts. Any new ORM model must inherit from `cv4.db.Base` and be imported in `alembic/env.py` for autogenerate to detect it.

## Training and Azure ML pipelines

Training runs both locally (via `cv4-train`) and as an Azure ML pipeline on the shared `lambda-0` Kubernetes cluster in the `team-CV4-2026` workspace. The pipeline is defined as reusable components under `components/` (`preprocess`, `train`, `evaluate`, `register`), with Conda environments in `configs/`.

Typical cloud workflow, driven by the scripts in `scripts/`:

```bash
uv sync --group azure
uv run python scripts/test_azure_connection.py       # verify workspace access
uv run python scripts/register_data_assets.py        # version the dataset
uv run python scripts/register_azure_environments.py # register Conda environments
uv run python scripts/submit_pipeline.py             # submit the full training pipeline
uv run python scripts/submit_sweep.py                # hyperparameter sweep
uv run python scripts/register_best_sweep_model.py   # register the best run's model
```

The preprocessing step crops plates from raw images and tiles them into non-overlapping 256 px patches; training then fits the Residual Attention U-Net with early stopping and learning-rate reduction, logging metrics and artefacts to MLflow. For deployment, `scripts/create_endpoint.py` and `scripts/canary_rollout.py` manage the managed endpoint that backs remote inference mode.

## Monitoring and drift detection

The `cv4.monitoring` package records prediction telemetry through OpenTelemetry into Azure Monitor (`telemetry.py`), maintains a confidence baseline (`baseline.py`), and flags distribution drift against that baseline (`drift.py`). The `drift_check` GitHub Actions workflow runs the check on a schedule, and the dashboard surfaces the current status at `GET /dashboard/drift-status`. Alert handling is documented in [`docs/alert_runbooks.md`](docs/alert_runbooks.md).

## Testing

The suite uses **pytest** with coverage reporting:

```bash
uv run pytest tests/                 # full suite
uv run pytest tests/test_api.py -v   # one module, verbose
uv run pytest --cov=cv4              # with coverage
```

`tests/test_db_connection.py` needs a reachable database and is skipped in CI. The CI pipeline (`.github/workflows/ci.yml`) runs Ruff lint and format checks, then the test suite with a coverage summary published to the job summary, on every push and pull request to `develop` and `main`.

## Development

```bash
uv sync --group dev
uv run pre-commit install   # enable the git hooks
```

- **Linting and formatting:** [Ruff](https://docs.astral.sh/ruff/) (line length 120), enforced by pre-commit and CI. Run manually with `uv run ruff check src/cv4 tests scripts` and `uv run ruff format src/cv4 tests scripts`.
- **Docstrings:** NumPy style, used throughout the package.
- **Branching:** work off `develop`; open pull requests into `develop`, which CI gates before changes reach `main`.
- **Docs:** the Sphinx site builds with `uv run sphinx-build -M html docs docs/_build` and deploys to GitHub Pages from `main`.

## Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| `Model file not found` / `Lane edges not found` | Weights are missing. Download `unet_256px.h5` and `lane_edges.npy` into `weights/`, or run `scripts/generate_lane_edges.py`. Check `CV4_MODEL_DIR`. |
| Server refuses to start, complaining about `JWT_SECRET_KEY` | The secret is required outside development. Set `JWT_SECRET_KEY`, or set `CV4_ENV=development` for local work. |
| `/predict` returns `401` | You are not authenticated. Log in first and send the cookies (`-b cookies.txt` with curl, or a `requests.Session`). |
| `/predict` returns `503` | The inference backend is not ready. In local mode the model failed to load; in remote mode the endpoint is unconfigured. Check `GET /health` and the logs. |
| `/predict` returns `507` | Not enough memory for inference. Reduce `BATCH_SIZE` in `config.py` or use a machine with more memory/GPU. |
| Database errors when calling `/feedback` | `DATABASE_URL` is unset or unreachable, or migrations have not run. Set the URL and run `alembic upgrade head`. |
| Docker `db` port conflict | A local Postgres is using `5433`. Stop it or change the host port mapping in `docker-compose.yml`. |
| `400` with "no petri dish detected" | The plate could not be located. Confirm the image is a full grayscale petri-dish scan in a supported format (PNG, TIFF, JPEG). |

## FAQ

**Why are there always five results even when fewer roots are found?**
Every plate has five lanes. Returning a fixed five-entry structure (with `has_root=false` for empty lanes) keeps client parsing simple. Always check `has_root` before reading coordinates.

**What is the difference between pixel and millimetre coordinates?**
The Python API returns both `root_tip_px` and `root_tip_mm`; the REST API returns `root_tip_px` plus the `crop` box. Pixel coordinates are in cropped-plate space — add the crop/bbox offset to reach original-image pixels. Millimetre coordinates use the physical plate size (150 mm edge).

**Local vs. remote inference — which should I use?**
Local mode is simplest and needs only the weights file. Remote mode offloads the U-Net to an Azure ML endpoint, useful when the API host has no GPU or you want to scale inference independently. Set `CV4_INFERENCE_MODE` accordingly and provide the `AZURE_ML_ENDPOINT_*` variables for remote.

**Do I need a database to run predictions?**
The pipeline and CLI do not. The API does, because `/predict` records a prediction (returning a `prediction_id`) and `/feedback` stores corrections. Use SQLite locally if you do not want to run PostgreSQL.

**Where do logs and outputs go?**
The CLI writes results and `cv4.log` to `outputs/` (gitignored). The API logs to standard output; telemetry goes to Azure Monitor when configured.

## Further documentation

- [`docs/pipeline_selection.md`](docs/pipeline_selection.md) — how the reference pipeline and model were chosen.
- [`docs/package_plan.md`](docs/package_plan.md) — package design and module plan.
- [`docs/alert_runbooks.md`](docs/alert_runbooks.md) — monitoring alerts and responses.
- [`docs/endpoint_performance.md`](docs/endpoint_performance.md) — endpoint performance notes.
- [`frontend/user_guide.md`](frontend/user_guide.md) — end-user guide for the web interface.
- Architecture diagrams (data, training, and deployment pipelines) are in [`docs/`](docs/).
- The full Sphinx documentation builds from `docs/` and is published to GitHub Pages.

## Authors

Developed by the ComputerVision4 team, Applied Data Science & AI, Breda University of Applied Sciences:
Ahmed Ali Alezabi, Abdelrahman Samara, Nazeer Idris, Stijn Korver, and Floris Lokhorst.
