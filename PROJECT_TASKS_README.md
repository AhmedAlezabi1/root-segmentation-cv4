# ComputerVision4 — Project Task Documentation

Block D (Engineer, MLOps): productionising and deploying our computer vision model (plant root segmentation for NPEC). This document records the work done for each issue, sprint by sprint, with the documentation and links to the deliverables.

- [ComputerVision4 — Project Task Documentation](#computervision4--project-task-documentation)
  - [Tasks](#tasks)
    - [Sprint 1 — Introduction, Briefing and Planning](#sprint-1--introduction-briefing-and-planning)
      - [Issue 1.1: Project Roadmap \& Milestones](#issue-11-project-roadmap--milestones)
      - [Issue 1.2: Pipeline Selection \& Package Structure Plan](#issue-12-pipeline-selection--package-structure-plan)
      - [Issue 1.3: GitHub Repository Setup \& Team Workflow](#issue-13-github-repository-setup--team-workflow)
      - [Issue 1.4: System Architecture Diagrams](#issue-14-system-architecture-diagrams)
    - [Sprint 2 — MVP Inference Application](#sprint-2--mvp-inference-application)
      - [Issue 2.1: Refactor Pipeline into Python Modules](#issue-21-refactor-pipeline-into-python-modules)
      - [Issue 2.2: CLI Tool](#issue-22-cli-tool)
      - [Issue 2.3: Unit Tests](#issue-23-unit-tests)
      - [Issue 2.4: Inference API](#issue-24-inference-api)
      - [Issue 2.5: Containerization \& On-Premise Deployment](#issue-25-containerization--on-premise-deployment)
      - [Issue 2.6: Combine README Fragments](#issue-26-combine-readme-fragments)
    - [Sprint 3 — Data Pipelines and Model Training in the Cloud](#sprint-3--data-pipelines-and-model-training-in-the-cloud)
      - [Issue 3.1: Azure ML Workspace \& Environment Setup](#issue-31-azure-ml-workspace--environment-setup)
      - [Issue 3.2: Versioned Data Assets](#issue-32-versioned-data-assets)
      - [Issue 3.3: Training \& Evaluation Pipeline](#issue-33-training--evaluation-pipeline)
      - [Issue 3.4: Hyperparameter Tuning](#issue-34-hyperparameter-tuning)
      - [Issue 3.5: Scheduled Retraining](#issue-35-scheduled-retraining)
      - [Issue 3.6: POST /feedback Endpoint](#issue-36-post-feedback-endpoint)
      - [Issue 3.7: CLI Training Command](#issue-37-cli-training-command)
      - [Issue 3.8: Frontend, Image Upload \& Prediction Display](#issue-38-frontend-image-upload--prediction-display)
      - [Issue 3.9: Frontend \& Database Containerization](#issue-39-frontend--database-containerization)
    - [Sprint 4 — Automated Deployment, Monitoring, and Continuous Retraining](#sprint-4--automated-deployment-monitoring-and-continuous-retraining)
      - [Issue 4.1: Azure ML Managed Endpoint](#issue-41-azure-ml-managed-endpoint)
      - [Issue 4.2: Deployment Strategies (Blue/Green or Canary)](#issue-42-deployment-strategies-bluegreen-or-canary)
      - [Issue 4.3: Cloud Deployment — Full Application](#issue-43-cloud-deployment--full-application)
      - [Issue 4.4: CI/CD Pipeline](#issue-44-cicd-pipeline)
      - [Issue 4.5: Operational Monitoring](#issue-45-operational-monitoring)
      - [Issue 4.6: Model Drift Detection](#issue-46-model-drift-detection)
      - [Issue 4.7: Stakeholder Monitoring Dashboard](#issue-47-stakeholder-monitoring-dashboard)
      - [Issue 4.8: User Feedback Correction Form](#issue-48-user-feedback-correction-form)
      - [Issue 4.9: JWT Multi-User Authentication](#issue-49-jwt-multi-user-authentication)
      - [Issue 4.10: Cloud Cost Analysis](#issue-410-cloud-cost-analysis)
      - [Issue 4.11: User-Facing Documentation (User Guide)](#issue-411-user-facing-documentation-user-guide)
  - [Team](#team)

## Tasks

### Sprint 1 — Introduction, Briefing and Planning

#### Issue 1.1: Project Roadmap & Milestones

Assigned to: Floris Lokhorst | ILO 4.4 | State: Done

##### 1.1 Documentation

To be written.

##### 1.1 Deliverables

- Link to project roadmap - ILO 4.4

#### Issue 1.2: Pipeline Selection & Package Structure Plan

Assigned to: Abdelrahman Samara | ILO 4.4 | State: Done

##### 1.2 Documentation

To be written.

##### 1.2 Deliverables

- Link to package structure plan - ILO 4.4

#### Issue 1.3: GitHub Repository Setup & Team Workflow

Assigned to: Nazeer Idris | ILO 4.4, ILO 9.3 | State: Done

##### 1.3 Documentation

To be written.

##### 1.3 Deliverables

- Link to repository / workflow documentation - ILO 4.4, ILO 9.3

#### Issue 1.4: System Architecture Diagrams

Assigned to: Ahmed Alezabi | ILO 4.4 | State: Done

##### 1.4 Documentation

To be written.

##### 1.4 Deliverables

- Link to architecture diagrams - ILO 4.4

### Sprint 2 — MVP Inference Application

#### Issue 2.1: Refactor Pipeline into Python Modules

Assigned to: Abdelrahman Samara | ILO 9.3 | State: Done

##### 2.1 Documentation

Takes a single grayscale petri-dish image and returns:
- Per-plant root tip coordinates in pixels and millimeters
- Confidence scores for each detection
- Segmentation masks for all 5 plants on the plate

Under the hood it runs a Residual Attention U-Net (256px patches with smooth blending and test-time augmentation), classical CV postprocessing (hysteresis thresholding, morphological closing, gap bridging), and graph-based pathfinding (Dijkstra over the root skeleton) to identify each plant's main root.

This package is the deployment-ready, productionized version of a research pipeline originally developed in Block B.

```
├── src/cv4/              # Main Python package
│   ├── config.py             # Central configuration, all tunable parameters
│   ├── preprocessing.py      # Load image, detect petri dish, crop, pad
│   ├── inference.py          # Load UNet model, run patch-based prediction
│   ├── postprocessing.py     # Probability map to clean binary mask
│   ├── segmentation.py       # Split mask into 5 per-plant instances
│   ├── localization.py       # Find main root and tip via Dijkstra pathfinding
│   ├── pipeline.py           # Orchestrator chaining all steps end-to-end
│   ├── cli.py                # Command-line entry point (cv4-predict)
│   ├── api/                  # FastAPI inference service
│   ├── db.py                 # SQLAlchemy engine and session factory
│   ├── data/                 # Data ingestion (Sprint 3+)
│   ├── training/             # Training and evaluation (Sprint 3+)
│   ├── models/               # Model architecture definitions
│   ├── pipelines/            # Azure ML pipeline definitions (Sprint 3+)
│   ├── monitoring/           # Drift detection and metrics (Sprint 4+)
│   └── utils/                # Shared logging and helpers
├── tests/                # Unit and integration tests
├── docker/               # Dockerfiles and docker-compose
├── frontend/             # Next.js web interface
├── docs/                 # Architecture diagrams, package plan, cost analysis
├── configs/              # Model and pipeline config files
└── scripts/              # One-off setup scripts (lane edges generation, etc.)
```

The pipeline data flow runs left-to-right: `preprocessing → inference → postprocessing → segmentation → localization`. Each module operates on numpy arrays in memory so no intermediate files are written between steps.

The main entry point is `cv4.pipeline.run_pipeline()`. It takes a path to a petri-dish image and returns a dictionary with the segmentation mask, per-plant masks, root tip coordinates, and timing info.

```python
from cv4 import pipeline

result = pipeline.run_pipeline("path/to/your/dish.png")

print(f"Processed in {result['processing_time_seconds']:.2f}s")
for plant in result["plants"]:
    if plant["has_root"]:
        tip_mm = plant["root_tip_mm"]
        print(f"Lane {plant['lane']}: tip at ({tip_mm['x']:.1f}, {tip_mm['y']:.1f}) mm "
              f"(confidence={plant['confidence']:.2f})")
    else:
        print(f"Lane {plant['lane']}: no root detected")
```

When `has_root` is `False`, the coordinate and confidence fields are explicitly `None` rather than zero. Always check `has_root` before reading them.

##### 2.1 Deliverables

- Link to cv4 package modules - ILO 9.3

#### Issue 2.2: CLI Tool

Assigned to: Ahmed Alezabi | ILO 9.3 | State: Done

##### 2.2 Documentation

The `cv4-predict` command runs the pipeline on a petri-dish image and outputs root tip coordinates, confidence scores, and optional visualizations.

```bash
uv run cv4-predict --image path/to/plant.png
```

Results are printed to the terminal and saved to the `outputs/` folder by default.

```bash
uv run cv4-predict --image path/to/plant.png --output results/my_results.json
```

```bash
uv run cv4-predict --image path/to/plant.png --visualize
```

The visualization shows the segmentation mask as a green overlay with red dots at each root tip and lane labels (L0 to L4).

```bash
uv run cv4-predict --image path/to/plant.png --model weights/custom_model.h5
```

| Option | Required | Description |
|--------|----------|-------------|
| `--image` | Yes | Path to the input plant image (PNG, JPG, or TIFF) |
| `--model` | No | Path to model weights (.h5 file). Defaults to config value |
| `--output` | No | Path to save results as JSON. Defaults to `outputs/` folder |
| `--visualize` | No | Save an image with the segmentation mask and root tips drawn on it |
| `--help` | No | Show help message and exit |

All output files are saved to the `outputs/` folder by default:

```
outputs/
    test_image_01_results.json
    test_image_01_results_viz.png
    cv4.log
```

The `outputs/` folder is gitignored.

| Error | Message |
|-------|---------|
| Image file not found | `Error: Image file not found: path/to/image.png` |
| Model file not found | `Error: Model file not found: path/to/model.h5` |
| Lane edges not found | `Error: Lane edges not found. Run scripts/generate_lane_edges.py` |
| Corrupt model file | `Error: Could not load model` |
| Out of memory | `Error: Not enough memory to predict patches. Try reducing batch_size` |

##### 2.2 Deliverables

- [Link to cli.py (cv4-predict command) - ILO 9.3](https://github.com/AhmedAlezabi1/root-segmentation-cv4/blob/feature-readme-links-/src/cv4/cli.py)

#### Issue 2.3: Unit Tests

Assigned to: Floris Lokhorst | ILO 9.3 | State: Done

##### 2.3 Documentation

To be written.

##### 2.3 Deliverables

- Link to tests directory - ILO 9.3

#### Issue 2.4: Inference API

Assigned to: Stijn Korver | ILO 9.5 | State: Done

##### 2.4 Documentation

The HTTP API is a thin layer over `cv4.pipeline.run_pipeline`. All algorithm logic lives in the pipeline; the API only handles auth, validation, and serialization.

The API is configured through environment variables. Variables can be set directly in the OS or in a `.env` file in the project root (read automatically at startup via `python-dotenv`).

| Variable | Required | Description |
|----------|----------|-------------|
| `CV4_API_KEYS` | Yes (non-dev) | Comma-separated list of valid API keys, e.g. `"key-prod-01,key-prod-02"` |
| `CV4_ENV` | No | Set to `development` to use the hardcoded default key when `CV4_API_KEYS` is not set |

The server refuses to start if `CV4_API_KEYS` is not set and `CV4_ENV` is not `development`. A deployment with no configured keys is a misconfiguration.

All requests to `POST /predict` require an API key in the `X-API-Key` header:

```
X-API-Key: your-api-key
```

Requests without a valid key receive `401 Unauthorized`. `GET /health` does not require authentication so orchestrators and load balancers can reach it.

**`POST /predict`** runs the segmentation pipeline on an uploaded image. Returns root tip coordinates and model confidence for each of the 5 plant lanes. The image must be a PNG, TIFF, or JPEG sent as `multipart/form-data` with field name `image`. Requests with missing or non-image content-type are rejected with `400` before any disk I/O.

```bash
curl -X POST http://localhost:8000/predict \
  -H "X-API-Key: admin" \
  -F "image=@plate_scan.png"
```

Response (200 OK):

```json
{
  "roots": [
    {
      "lane": 0,
      "has_root": true,
      "root_tip_px": { "x": 145, "y": 890 },
      "confidence": 0.9123
    },
    { "lane": 1, "has_root": false, "root_tip_px": null, "confidence": null },
    { "lane": 2, "has_root": false, "root_tip_px": null, "confidence": null },
    { "lane": 3, "has_root": false, "root_tip_px": null, "confidence": null },
    { "lane": 4, "has_root": false, "root_tip_px": null, "confidence": null }
  ],
  "processing_time_s": 4.21
}
```

**`GET /health`** is the liveness and readiness check. Returns `200` when the model is loaded and ready, `503` when the model failed to load at startup. Container orchestrators use this endpoint to gate traffic routing.

```json
{ "status": "ok", "model_loaded": true }
```

All errors use the same envelope:

```json
{ "detail": "Human-readable description of the error." }
```

| Code | Cause |
|------|-------|
| `400 Bad Request` | Missing or non-image content-type, image could not be decoded, or no petri dish detected |
| `401 Unauthorized` | Missing or invalid `X-API-Key` header |
| `500 Internal Server Error` | Unexpected error during inference, check server logs |
| `503 Service Unavailable` | Model failed to load at startup, check `/health` and server logs |
| `507 Insufficient Storage` | Insufficient GPU or CPU memory to run inference |

FastAPI generates interactive docs automatically: Swagger UI at `http://localhost:8000/docs` and ReDoc at `http://localhost:8000/redoc`.

##### 2.4 Deliverables

- Link to api/app.py - ILO 9.5

#### Issue 2.5: Containerization & On-Premise Deployment

Assigned to: Nazeer Idris | ILO 9.5 | State: Done

##### 2.5 Documentation

Three long-running services come up together: the FastAPI backend, the Next.js frontend, and a Postgres database, plus a one-shot `db-migrate` container that applies Alembic migrations before the backend starts. All four are wired on a private network. The same backend image promotes to on-prem and to Azure, so nothing about it is dev-specific. Auth, model weights, and database credentials are injected into the backend at runtime, never baked in. The one exception is the frontend's backend URL: Next.js bakes proxy rewrites at build time, so it's passed as a build argument.

From the repo root:

```bash
docker compose -f docker/docker-compose.yml --env-file docker/.env up --build
```

Once up:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API docs (Swagger): http://localhost:8000/docs
- Postgres: localhost:5433 (mapped off 5432 to avoid clashing with a local Postgres install)

Frontend talks to the backend, the backend talks to the database, Postgres talks to no one. They share a private bridge network called `cv4_net`, which lets each service reach the others by service name (`db`, `backend`) instead of `localhost`. Startup order is enforced by `depends_on` with health and completion conditions. Postgres starts first; once it reports healthy via `pg_isready`, the one-shot `db-migrate` container runs `alembic upgrade head` against it and exits 0. Only then does the backend start. The frontend waits for the backend. This guarantees the schema is always at head before any service tries to query it. The backend image does not contain model weights. They're mounted in at runtime from `weights/` at the repo root into `/app/models` inside the container, read-only.

For on-premise deployment, the cv4 stack is deployed to the BUas Portainer server (`portainer.example.com`). Images are pre-built and hosted on GHCR; Portainer pulls them at deploy time.

| Service | GHCR image |
|---------|-----------|
| backend | `ghcr.io/bredauniversityadsai/2025-26d-fai2-adsai-group-computervision4/cv4-backend:latest` |
| frontend | `ghcr.io/bredauniversityadsai/2025-26d-fai2-adsai-group-computervision4/cv4-frontend:latest` |

Images are built and pushed automatically by `.github/workflows/docker-publish.yml` on every merge to `main`.

| Check | Expected |
|-------|----------|
| `GET http://portainer.example.com:2017/health` | `200 OK` |
| `POST http://portainer.example.com:2017/predict` with a test image | Valid prediction JSON |
| `http://portainer.example.com:2016` in a browser | Frontend loads and the status badge reads "model ready" |

##### 2.5 Deliverables

- Link to docker folder (Dockerfiles, compose files) - ILO 9.5

#### Issue 2.6: Combine README Fragments

Assigned to: Abdelrahman Samara | ILO 9.3 | State: Done

##### 2.6 Documentation

The separate documentation fragments written by each team member were merged into a single, comprehensive developer-facing `README.md` covering the package, Python usage, the CLI, the REST API, the Docker stack, and the on-premise deployment.

##### 2.6 Deliverables

- Link to main README.md - ILO 9.3

### Sprint 3 — Data Pipelines and Model Training in the Cloud

#### Issue 3.1: Azure ML Workspace & Environment Setup

Assigned to: Floris Lokhorst | ILO 8.9 | State: Done

##### 3.1 Documentation

To be written.

##### 3.1 Deliverables

- Link to environment setup - ILO 8.9

#### Issue 3.2: Versioned Data Assets

Assigned to: Ahmed Alezabi | ILO 9.4 | State: Done

##### 3.2 Documentation

This module handles uploading and versioning the CV4 plant root segmentation dataset in Azure Machine Learning. Each dataset split (train, validation) is stored as a single versioned Data Asset containing both images and their corresponding segmentation masks. Bundling images and masks together ensures pairs are always in sync and prevents version skew.

| Asset Name      | Contents                                     | Description                       |
|-----------------|----------------------------------------------|-----------------------------------|
| `cv4-raw-train` | `train_images/train/` + `train_masks/train/` | Training split (images + masks)   |
| `cv4-raw-val`   | `val_images/val/` + `val_masks/val/`         | Validation split (images + masks) |

Both assets live in the `workspaceblobstore` of the `team-CV4-2026` Azure ML workspace.

| Version | Contents                              |
|---------|---------------------------------------|
| `v1`    | Sample subset (10 pairs per split)    |
| `v2`    | Full dataset (475 train pairs)        |

**Upload:**
```bash
uv run --group azure python scripts/register_data_assets.py \
    --base-path data/dataset
```

| Argument      | Required | Description                                          |
|---------------|----------|------------------------------------------------------|
| `--base-path` | Yes      | Path to folder containing `train/` and `val/`        |
| `--dry-run`   | No       | Preview the upload without making changes            |

**Behaviour:**
- Validates that all expected subfolders exist before uploading.
- Uploads each split as a `URI_FOLDER` Data Asset.
- Azure auto-increments the version on each upload; previous versions are preserved.
- Exits with code 1 if any upload fails (CI-friendly).
- Logs progress and errors with timestamps.

```python
from azure.ai.ml import MLClient

ml_client = MLClient(...)
train_data = ml_client.data.get("cv4-raw-train", version="2")
```

**Why 2 assets instead of 4?**
Images and masks for the same split are bundled into a single asset so they are always versioned together. This removes the risk of uploading new images while forgetting to update the corresponding masks.

**Note on deletion:**
Data assets created with the Azure ML v2 SDK cannot be deleted, only archived or up-versioned. The initial 4-asset sample upload was archived after restructuring to 2 assets.

##### 3.2 Deliverables

- [Link to register_data_assets.py - ILO 9.4](https://github.com/AhmedAlezabi1/root-segmentation-cv4/blob/feature-readme-links-/scripts/register_data_assets.py)


#### Issue 3.3: Training & Evaluation Pipeline

Assigned to: Abdelrahman Samara | ILO 8.9 | State: Done

##### 3.3 Documentation

To be written.

##### 3.3 Deliverables

- Link to training pipeline - ILO 8.9

#### Issue 3.4: Hyperparameter Tuning

Assigned to: Abdelrahman Samara | ILO 8.9 | State: Done

##### 3.4 Documentation

To be written.

##### 3.4 Deliverables

- Link to hyperparameter tuning pipeline - ILO 8.9

#### Issue 3.5: Scheduled Retraining

Assigned to: Floris Lokhorst | ILO 9.5 | State: Done

##### 3.5 Documentation

To be written.

##### 3.5 Deliverables

- Link to scheduled retraining setup - ILO 9.5

#### Issue 3.6: POST /feedback Endpoint

Assigned to: Stijn Korver | ILO 9.5 | State: Done

##### 3.6 Documentation

To be written.

##### 3.6 Deliverables

- Link to feedback endpoint - ILO 9.5

#### Issue 3.7: CLI Training Command

Assigned to: Ahmed Alezabi | ILO 9.3 | State: Done

##### 3.7 Documentation

The `cv4-train` command trains the Residual Attention U-Net on paired image/mask data and saves the trained weights. The same `cv4.training.train` module powers both this local CLI and the cloud Azure ML pipeline, so behaviour is identical in both contexts; only the compute and data source differ. Locally the command is intended for quick validation via the smoke test, while full training runs on the Azure GPU cluster.

The model trains on 256×256 patches, not full petri-dish images. Raw images are too large and too sparse (most of the frame is background) to train on directly, so they must be preprocessed into patches first. Preprocessing crops the petri dish out of each raw image and tiles the cropped region into non-overlapping patches, producing far more training examples from the same source images.

```bash
uv run python -m cv4.training.preprocess \
    --train-data-path data/dataset/train \
    --val-data-path data/dataset/val \
    --output-path data/patched_dataset \
    --patch-size 256 \
    --mask-suffix "_root_mask.tif"
```

```bash
# Smoke test: caps training at 2 steps per epoch for a quick end-to-end check
uv run cv4-train --data data/patched_dataset --smoke-test

# Full training run
uv run cv4-train --data data/patched_dataset

# Custom run with MLflow tracking
uv run cv4-train --data data/patched_dataset \
    --epochs 30 \
    --batch-size 16 \
    --augmentation mild \
    --experiment-name "cv4-segmentation" \
    --run-name "baseline_mild_30ep"
```

The smoke test exists so the full preprocess-then-train chain can be verified on any machine in under a minute, without committing to a full training run. It caps the steps per epoch but still exercises every part of the code path: data loading, augmentation, the model forward/backward pass, MLflow logging, and model saving.

| Option | Required | Description |
|--------|----------|-------------|
| `--data` | Yes | Path to the patched dataset root with the four-folder layout |
| `--model-output` | No | Where to save the trained model (.h5). Defaults to `outputs/` |
| `--epochs` | No | Maximum training epochs; early stopping may halt sooner (default 50) |
| `--batch-size` | No | Number of patches per training step (default 32) |
| `--learning-rate` | No | Initial learning rate for the optimizer (default 1e-4) |
| `--augmentation` | No | Augmentation preset: `none`, `mild`, or `strong` (default `mild`) |
| `--experiment-name` | No | MLflow experiment name for tracking |
| `--run-name` | No | MLflow run name to identify this run in the UI |
| `--smoke-test` | No | Quick run capped at 2 steps per epoch for end-to-end validation |
| `--help` | No | Show help message and exit |

Training uses early stopping and learning-rate reduction to prevent overfitting, and logs hyperparameters plus per-epoch metrics to MLflow. All failures exit with code 1 so callers and CI can detect them.

##### 3.7 Deliverables

- [Link to cli.py (cv4-train command) - ILO 9.3](https://github.com/AhmedAlezabi1/root-segmentation-cv4/blob/feature-readme-links-/src/cv4/cli.py)

#### Issue 3.8: Frontend, Image Upload & Prediction Display

Assigned to: Stijn Korver | ILO 10.3 | State: Done

##### 3.8 Documentation

To be written.

##### 3.8 Deliverables

- Link to frontend - ILO 10.3

#### Issue 3.9: Frontend & Database Containerization

Assigned to: Nazeer Idris | ILO 9.5 | State: Done

##### 3.9 Documentation

To be written.

##### 3.9 Deliverables

- Link to frontend and database containerization - ILO 9.5

### Sprint 4 — Automated Deployment, Monitoring, and Continuous Retraining

#### Issue 4.1: Azure ML Managed Endpoint

Assigned to: Ahmed Alezabi | ILO 9.5 | State: Done

##### 4.1 Documentation

The registered model `cv4_root_segmentation` was deployed as a live Azure ML online endpoint that serves predictions over a URL, the cloud equivalent of the FastAPI backend container. A scoring script (`score.py`) loads the model on startup (`init()`), turns on GPU memory growth so the endpoint does not reserve the whole shared GPU, and on each request decodes a base64 image, runs the cv4 pipeline, and returns the per-lane root predictions as JSON matching the local `/predict` schema. Because the registered model contains only `model.h5`, the required `lane_edges.npy` is bundled alongside the scoring script. A dedicated `cv4-inference` environment was built (GPU TensorFlow plus the CV packages plus `azureml-inference-server-http`), since the existing environments could not serve. The deployment uses key-based authentication, runs on a GPU instance, is configured with autoscaling (min 1, max 3 instances), and routes 100% of traffic to the deployment. A Kubernetes online endpoint on the lambda-0 cluster was used because the hub-based workspace's compute is lambda-0; the concept is identical to a managed endpoint. Latency and throughput were measured (about 5 to 7 seconds per warm request, about 10 requests per minute on one instance) and documented. The endpoint is taken down when not in use to free the shared GPU and can be recreated from the scripts at any time.

###### How to deploy, call, test, and take down the endpoint

**Before you start.** You need Azure access and the azure dependencies:

```bash
az login
uv sync --group azure
```

The endpoint runs on the lambda-0 Kubernetes cluster. Its IP is only reachable from the BUas network or over the BUas VPN, so if you are off campus, turn on the VPN before calling it.

**Important: the endpoint is usually taken down.** The endpoint uses a GPU, and the GPU is shared across all teams (only a few are available). So we take the endpoint down when we are not using it. This means: before anyone can call it, it has to be deployed; each time it is deployed it gets a fresh URL and a fresh key; and after you finish, take it down again to free the GPU.

**Step 1: Deploy the endpoint.** This creates the endpoint, loads the model, sets autoscaling (min 1, max 3 instances), and routes 100% of traffic to it:

```bash
uv run --group azure python scripts/create_endpoint.py
```

When it finishes, it prints the scoring URL and the key. The key is a secret: do not commit it or post it publicly. If the serving environment does not exist yet, register it first with `scripts/register_inference_env.py`.

**Step 2: Call the endpoint.** It expects a base64-encoded image inside a JSON body, and it needs the key for authentication.

Request format:

```json
{ "image": "<base64-encoded image bytes>" }
```

Response format (always 5 entries, one per lane; check `has_root` before reading the other fields):

```json
{
  "roots": [
    { "lane": 0, "has_root": true,
      "root_tip_px": { "x": 619, "y": 1958 }, "confidence": 0.99 },
    { "lane": 4, "has_root": false, "root_tip_px": null, "confidence": null }
  ],
  "processing_time_s": 5.6
}
```

Example call from Python:

```python
import base64
import requests

# The URL and key are printed by create_endpoint.py when you deploy.
SCORING_URL = "https://cv4-root-seg.westeurope.inference.ml.azure.com/score"
API_KEY = "your-key-here"

# Read the image and turn it into a base64 string.
with open("plant.png", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode("utf-8")

# Send the request.
response = requests.post(
    SCORING_URL,
    headers={
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    },
    json={"image": image_b64},
)

result = response.json()
for plant in result["roots"]:
    if plant["has_root"]:
        print(f"Lane {plant['lane']}: tip at {plant['root_tip_px']}, "
              f"confidence {plant['confidence']}")
    else:
        print(f"Lane {plant['lane']}: no root detected")
```

The first request after a fresh deploy is slow (about 17 to 20 seconds) while the worker warms up. After that, requests take about 5 to 7 seconds.

**Step 3: Test it with the ready-made script.** This sends a real image to the live endpoint and prints the prediction, so you do not have to write the calling code yourself. The `--from-workspace` flag fetches the URL and key automatically:

```bash
uv run --group azure python scripts/test_endpoint_azure.py --image path/to/plant.png --from-workspace
```

To measure speed (latency and throughput):

```bash
uv run --group azure python scripts/measure_endpoint.py --image path/to/plant.png --requests 5 --from-workspace
```

**Step 4: Take the endpoint down.** When done, set traffic to 0 (you cannot delete a deployment that has traffic), then delete the endpoint:

```bash
# set traffic to 0
uv run --group azure python -c "from azure.ai.ml import MLClient; from azure.identity import DefaultAzureCredential; from cv4.config import *; c=MLClient(DefaultAzureCredential(), AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP, AZURE_WORKSPACE_NAME); e=c.online_endpoints.get('cv4-root-seg'); e.traffic={'blue':0}; c.online_endpoints.begin_create_or_update(e).result(); print('traffic 0')"

# delete the endpoint
uv run --group azure python -c "from azure.ai.ml import MLClient; from azure.identity import DefaultAzureCredential; from cv4.config import *; c=MLClient(DefaultAzureCredential(), AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP, AZURE_WORKSPACE_NAME); c.online_endpoints.begin_delete(name='cv4-root-seg').result(); print('deleted')"
```

To bring it back later, just run `create_endpoint.py` again.

**Connecting the endpoint to the backend.** If you want the backend to call this endpoint instead of running the model itself, the backend makes the same request shown in the Python example above: a POST to the scoring URL, with the key in the Authorization header and a base64 image in the body. The URL and key should be passed to the backend as environment variables, never hardcoded.

##### 4.1 Deliverables
##### 4.1 Deliverables

- [score.py](https://github.com/AhmedAlezabi1/root-segmentation-cv4/blob/feature-readme-links-/src/cv4/deployment/score.py) and [lane_edges.npy](https://github.com/AhmedAlezabi1/root-segmentation-cv4/blob/feature-readme-links-/src/cv4/deployment/lane_edges.npy) - ILO 9.5
- [create_endpoint.py](https://github.com/AhmedAlezabi1/root-segmentation-cv4/blob/feature-readme-links-/scripts/create_endpoint.py) - ILO 9.5
- [register_inference_env.py](https://github.com/AhmedAlezabi1/root-segmentation-cv4/blob/feature-readme-links-/scripts/register_inference_env.py) and [env-inference.yml](https://github.com/AhmedAlezabi1/root-segmentation-cv4/blob/feature-readme-links-/configs/env-inference.yml) - ILO 9.5
- [test_endpoint_azure.py](https://github.com/AhmedAlezabi1/root-segmentation-cv4/blob/feature-readme-links-/scripts/test_endpoint_azure.py) and [measure_endpoint.py](https://github.com/AhmedAlezabi1/root-segmentation-cv4/blob/feature-readme-links-/scripts/measure_endpoint.py) - ILO 9.5
- [endpoint_performance.md](https://github.com/AhmedAlezabi1/root-segmentation-cv4/blob/feature-readme-links-/docs/endpoint_performance.md) - ILO 9.5

#### Issue 4.2: Deployment Strategies (Blue/Green or Canary)

Assigned to: Abdelrahman Samara | ILO 9.5 | State: To Do

##### 4.2 Documentation

To be written.

##### 4.2 Deliverables

- Link to deployment strategy work - ILO 9.5

#### Issue 4.3: Cloud Deployment — Full Application

Assigned to: Floris Lokhorst | ILO 9.5 | State: Doing

##### 4.3 Documentation

To be written.

##### 4.3 Deliverables

- Link to cloud deployment - ILO 9.5

#### Issue 4.4: CI/CD Pipeline

Assigned to: Abdelrahman Samara | ILO 9.5 | State: Doing

##### 4.4 Documentation

To be written.

##### 4.4 Deliverables

- Link to CI/CD workflows - ILO 9.5

#### Issue 4.5: Operational Monitoring

Assigned to: Ahmed Alezabi | ILO 9.5 | State: Doing

##### 4.5 Documentation

Operational monitoring was added so the team can see the technical health of the deployed system (request rates, response times, error rates, and browser activity) and be alerted automatically when something goes wrong. The monitoring uses Azure Application Insights, with a single resource named `cv4-monitoring` collecting telemetry from both the backend and the frontend, plus Azure Monitor alerts and an operational dashboard built on top of that data.

The work was done in four parts: backend monitoring, frontend monitoring, alerts, and a dashboard.

###### Backend monitoring

The FastAPI backend was instrumented with the Azure Monitor OpenTelemetry distribution. A small module (`src/cv4/monitoring/telemetry.py`) provides a `setup_monitoring(app)` function that reads the connection string from the `APPLICATIONINSIGHTS_CONNECTION_STRING` environment variable and, if it is present, configures Azure Monitor and auto-instruments FastAPI. Auto-instrumentation means every HTTP request is tracked automatically (its route, status code, and duration) without adding code to each endpoint. The module also provides a `track_event` helper, which is used in the predict endpoint to log a custom `prediction_succeeded` event. If the connection string is not set (for example in local development or tests), monitoring is skipped with a warning and the application runs normally, so monitoring can never crash the app.

Adding the Azure monitoring packages caused a dependency conflict, because they share the OpenTelemetry library with MLflow, which the training pipeline depends on. This was resolved by pinning a version combination (`azure-monitor-opentelemetry` 1.8.8, which resolves `opentelemetry-sdk` to 1.40.0) that satisfies both, and this was verified by confirming MLflow still imports and runs after the change. The change to the shared dependency files was documented in the pull request so the team was aware.

The backend monitoring was tested locally by running the backend with the connection string set, sending requests, and confirming they appeared in `cv4-monitoring` in Azure (request count, failed request count, and response time all tracked correctly).

###### Frontend monitoring

The Next.js frontend was instrumented with the Application Insights JavaScript SDK. A setup module (`frontend/app/lib/telemetry.ts`) initialises the SDK with automatic page-view tracking, browser exception tracking, and tracking of the calls the browser makes to the backend. A small client component (`frontend/app/lib/TelemetryProvider.tsx`) runs the setup once when the app loads, and it is added to the root layout so it runs on every page. The connection string is read from the `NEXT_PUBLIC_APPLICATIONINSIGHTS_CONNECTION_STRING` environment variable. A browser connection string is necessarily visible to the client, which is normal and expected for client-side monitoring.

The frontend monitoring was tested locally by browsing several pages and confirming the page views appeared in `cv4-monitoring` in Azure, alongside the backend requests, so the full picture (frontend page views plus backend requests) is visible in one place.

###### Alerts

Alert rules were created in Azure Monitor on the `cv4-monitoring` data, each notifying the team by email through an action group named `cv4-monitoring-alerts`:

- `cv4-high-error-rate`: fires when there are more than 5 failed requests in a 5-minute window.
- `cv4-high-latency`: fires when the average server response time is over 10 seconds in a 5-minute window.
- `cv4-server-exceptions`: fires when there are more than 3 server exceptions in a 5-minute window.
- `Failure Anomalies - cv4-monitoring`: an Application Insights smart-detector alert that uses machine learning to spot unusual rises in the failure rate.

A short runbook for each alert (what it means, likely causes, how to investigate, and how to resolve) is documented in `docs/alert_runbooks.md`. A fire test was attempted by generating 404 errors, but Azure's failed-request metric counts only server-side 5xx errors, so the threshold was not crossed; the rules are confirmed enabled and correctly configured.

###### Operational dashboard

A shared Azure dashboard named `cv4-operational-dashboard` was created for the engineering team, showing five technical metrics on one screen: server requests, failed requests, server response time, server exceptions, and availability. This gives an at-a-glance view of system health and is distinct from the stakeholder dashboard in Issue 4.7, which is aimed at non-technical users.

###### Platform note and going live

The live applications run on the on-premise Portainer server, not on Azure Container Apps. Azure cannot collect container infrastructure metrics (CPU, memory, restarts) for containers it does not host, so those specific metrics are not available on this platform. The ILO requirement for technical application monitoring is still met, because the application telemetry (requests, errors, latency, page views, custom events) is sent from the Portainer-hosted apps to Application Insights over the internet and works regardless of where the apps are hosted. To enable monitoring on the live server, the backend and frontend on Portainer need their respective connection-string environment variables set and a redeploy; the code is written to disable monitoring safely if the variable is absent, so an incomplete rollout cannot crash the live app.

##### 4.5 Deliverables

- [Backend telemetry module (telemetry.py)](https://github.com/AhmedAlezabi1/root-segmentation-cv4/blob/feature-readme-links-/src/cv4/monitoring/telemetry.py) - ILO 9.5
- [Backend monitoring integration (app.py)](https://github.com/AhmedAlezabi1/root-segmentation-cv4/blob/feature-readme-links-/src/cv4/api/app.py) - ILO 9.5
- [Frontend telemetry (telemetry.ts)](https://github.com/AhmedAlezabi1/root-segmentation-cv4/blob/feature-readme-links-/frontend/app/lib/telemetry.ts) and [TelemetryProvider.tsx](https://github.com/AhmedAlezabi1/root-segmentation-cv4/blob/feature-readme-links-/frontend/app/lib/TelemetryProvider.tsx) - ILO 9.5
- [Alert runbooks (alert_runbooks.md)](https://github.com/AhmedAlezabi1/root-segmentation-cv4/blob/feature-readme-links-/docs/alert_runbooks.md) - ILO 9.5
- cv4-operational-dashboard (in the Azure portal, under Dashboards) - ILO 9.5

#### Issue 4.6: Model Drift Detection

Assigned to: Nazeer Idris | ILO 9.5 | State: Doing

##### 4.6 Documentation

To be written.

##### 4.6 Deliverables

- Link to drift detection - ILO 9.5

#### Issue 4.7: Stakeholder Monitoring Dashboard

Assigned to: Stijn Korver | ILO 10.3 | State: Done

##### 4.7 Documentation

To be written.

##### 4.7 Deliverables

- Link to stakeholder dashboard - ILO 10.3

#### Issue 4.8: User Feedback Correction Form

Assigned to: Stijn Korver | ILO 10.3 | State: Done

##### 4.8 Documentation

To be written.

##### 4.8 Deliverables

- Link to feedback correction form - ILO 10.3

#### Issue 4.9: JWT Multi-User Authentication

Assigned to: Stijn Korver | ILO 9.5 | State: Done

##### 4.9 Documentation

To be written.

##### 4.9 Deliverables

- Link to JWT authentication - ILO 9.5

#### Issue 4.10: Cloud Cost Analysis

Assigned to: Nazeer Idris | ILO 4.4 | State: To Do

##### 4.10 Documentation

To be written.

##### 4.10 Deliverables

- Link to cost analysis - ILO 4.4

#### Issue 4.11: User-Facing Documentation (User Guide)

Assigned to: Stijn Korver | ILO 10.3 | State: To Do

##### 4.11 Documentation

To be written.

##### 4.11 Deliverables

- Link to user guide - ILO 10.3

## Team

- Ahmed Alezabi (245226)
- Abdelrahman Samara (245769)
- Nazeer Idris (244968)
- Stijn Korver (242824)
- Floris Lokhorst (236918)
