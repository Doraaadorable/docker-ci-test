# Digit classifier: from model to container to Kubernetes

A small, complete deployment path for a machine-learning model: train once, package the model in a hardened container, test it in CI, publish only the image that passed its tests, and run it on Kubernetes with health probes and rolling updates. The model is deliberately simple (logistic regression on scikit-learn's 8×8 digits, about 96% test accuracy) so the focus stays on the path, which can carry any scikit-learn model.

```
train.py ──► model/ (model, metadata, held-out set) ──► Docker image (gunicorn, non-root)
                                                          │
GitHub Actions: tests ─► build + smoke test ─► push sha-<commit> ─► re-test pushed image ─► tag latest
                                                          │
                                         Kubernetes: 3 replicas, readiness/liveness probes
```

## API

| Endpoint | Returns |
|---|---|
| `GET /health` | `{"status": "ok", "test_accuracy": ..., "pod": <hostname>}`; used by the Docker and Kubernetes health checks |
| `GET /metadata` | Model type, training and test sizes, accuracy, seed, data hash, scikit-learn version |
| `GET /sample/<i>` | Held-out test image `i` with its label (404 if out of range) |
| `POST /predict` | `{"pixels": [64 numbers in 0–16]}` → `{"prediction": 7, "probability": 0.98}`; invalid input → 400 with a message |
| `GET /visits` | Visit counter in Postgres (Compose only); 503 if no database |

## Run it

**Python**

```bash
pip install -r requirements-dev.txt
python train.py
python -m pytest -q tests
python app.py
```

**Code checks**

```bash
ruff check .            # lint: unused code, naming, missing type hints and docstrings
ruff format --check .   # formatting
mypy                    # static type check (strict)
```

Every function declares its argument and return types. The rules are in `pyproject.toml`.

**Docker**

```bash
docker build -t digits-classifier .
docker run --rm -p 5050:5050 --read-only --cap-drop ALL digits-classifier
scripts/smoke_test.sh
```

**Compose (with Postgres)**

```bash
docker compose up --build
curl localhost:5050/visits
```

**Kubernetes (Minikube)**

To run a local build instead of the published image, load it with `minikube image load digits-classifier` and point the Deployment at `digits-classifier` with `imagePullPolicy: IfNotPresent`.

```bash
kubectl apply -f k8s/deployment.yaml
kubectl get pods -l app=digits-classifier
minikube service digits-classifier --url
```

Then try the behaviour Kubernetes adds:

```bash
kubectl delete pod <one-pod>
kubectl get pods -w
kubectl scale deployment/digits-classifier --replicas=6
kubectl set image deployment/digits-classifier digits-classifier=ghcr.io/doraaadorable/docker-ci-test:sha-<commit>
kubectl rollout undo deployment/digits-classifier
```

Deleting a pod makes the Deployment replace it; scaling adds replicas; the last two commands roll forward to a specific build and back again. A pod receives traffic only once its readiness probe passes, and the rollout keeps every existing replica serving until its replacement is ready (`maxUnavailable: 0`).

## Design choices

| Choice | Reason |
|---|---|
| Train at build time (`RUN python train.py`) | Every container serves the same model; pods start in seconds instead of retraining |
| Fully pinned `requirements.txt` | Rebuilding the same commit installs the same versions |
| Dependencies copied before code | Code changes reuse the cached dependency layer |
| gunicorn, non-root user (uid 10001), `HEALTHCHECK` | Production server, least privilege, self-reported health |
| Container runs with a read-only filesystem and no capabilities | Tested this way in CI and enforced in the Kubernetes `securityContext` |
| Input validation with JSON errors | Bad requests get a 400 that explains the problem instead of a 500 |
| Publish `sha-<commit>`, test the pushed image, then tag `latest` | `latest` always points at an image that passed the smoke test after publishing |
| Probes, resource requests and limits | Kubernetes routes traffic only to ready pods, restarts stuck ones and schedules by real needs |

## CI (`.github/workflows/ci.yml`)

1. **test**: unit tests (16), covering reproducible training, every endpoint, and nine kinds of invalid input.
2. **lint**: `ruff check`, `ruff format --check` and `mypy` in strict mode.
3. **manifests**: `kubeconform -strict` validates `k8s/` against the Kubernetes API schema.
4. **image**: builds the image, runs it read-only with no capabilities, and runs `scripts/smoke_test.sh` (health, a real prediction, a rejected bad request).
5. **publish** (pushes to `main`): builds for amd64 and arm64, pushes `sha-<commit>` to GitHub Container Registry, smoke-tests the pushed image, then tags it `latest`.

## Files

| Path | Contents |
|---|---|
| `train.py` | Trains the model and writes `model/` (model, metadata, held-out set) |
| `app.py` | Flask service |
| `tests/test_app.py` | Unit tests |
| `pyproject.toml` | Ruff and mypy configuration |
| `scripts/smoke_test.sh` | Checks a running container |
| `Dockerfile`, `.dockerignore` | Image definition |
| `docker-compose.yml` | App plus Postgres |
| `k8s/deployment.yaml` | Deployment and NodePort Service |
