"""Digit-classifier service: serves a model trained by train.py."""

import json
import os
import socket
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from flask import Flask, Response, jsonify, request
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from werkzeug.exceptions import HTTPException

ErrorResponse = tuple[Response, int]

MODEL_DIR = Path(os.environ.get("MODEL_DIR", Path(__file__).resolve().parent / "model"))

app = Flask(__name__)


@lru_cache(maxsize=1)
def artifacts() -> tuple[LogisticRegression, dict[str, Any], NDArray[np.float64], NDArray[np.int_]]:
    """Load the model, its metadata and the held-out test set once per process."""
    model = joblib.load(MODEL_DIR / "model.joblib")
    metadata = json.loads((MODEL_DIR / "metadata.json").read_text())
    test = np.load(MODEL_DIR / "test_set.npz")
    return model, metadata, test["X"], test["y"]


def error(status: int, message: str) -> ErrorResponse:
    """Build a JSON error body with the given HTTP status."""
    return jsonify(error=message), status


@app.errorhandler(HTTPException)
def http_error(exc: HTTPException) -> ErrorResponse:
    """Return Flask's own HTTP errors (404, 405, ...) as JSON instead of HTML."""
    return error(exc.code or 500, exc.description or exc.name)


@app.get("/")
def home() -> str:
    """Landing page showing the model's test accuracy."""
    _, metadata, _, _ = artifacts()
    return f"<h1>Digit classifier</h1><p>Test accuracy: {metadata['test_accuracy']:.1%}</p>"


@app.get("/health")
def health() -> Response:
    """Health check used by Docker and the Kubernetes probes."""
    _, metadata, _, _ = artifacts()
    return jsonify(status="ok", test_accuracy=metadata["test_accuracy"], pod=socket.gethostname())


@app.get("/metadata")
def model_metadata() -> Response:
    """Return the metadata written by train.py."""
    _, metadata, _, _ = artifacts()
    return jsonify(metadata)


@app.get("/sample/<int:i>")
def sample(i: int) -> Response | ErrorResponse:
    """Return held-out test image `i` with its label."""
    _, _, X_test, y_test = artifacts()
    if not 0 <= i < len(y_test):
        return error(404, f"sample index must be between 0 and {len(y_test) - 1}")
    return jsonify(pixels=X_test[i].tolist(), label=int(y_test[i]))


@app.post("/predict")
def predict() -> Response | ErrorResponse:
    """Validate the posted pixels and return the predicted digit with its probability."""
    model, metadata, _, _ = artifacts()
    body = request.get_json(silent=True)
    if not isinstance(body, dict) or "pixels" not in body:
        return error(400, 'send JSON like {"pixels": [64 numbers]}')
    pixels = body["pixels"]
    n, (low, high) = metadata["n_features"], metadata["pixel_range"]
    if not isinstance(pixels, list) or len(pixels) != n:
        return error(400, f"pixels must be a list of {n} numbers")
    if not all(isinstance(p, (int, float)) and not isinstance(p, bool) for p in pixels):
        return error(400, "pixels must be numbers")
    x = np.asarray(pixels, dtype=float)
    if not np.all(np.isfinite(x)) or x.min() < low or x.max() > high:
        return error(400, f"pixel values must be between {low} and {high}")
    probabilities = model.predict_proba(x.reshape(1, -1))[0]
    best = int(np.argmax(probabilities))
    return jsonify(prediction=int(model.classes_[best]), probability=round(float(probabilities[best]), 4))


@app.get("/visits")
def visits() -> Response | ErrorResponse:
    """Count page visits in Postgres (used by docker-compose.yml)."""
    try:
        import psycopg2

        con = psycopg2.connect(
            host=os.environ.get("DB_HOST", "localhost"),
            dbname=os.environ.get("DB_NAME", "postgres"),
            user=os.environ.get("DB_USER", "postgres"),
            password=os.environ.get("DB_PASSWORD", ""),
            connect_timeout=3,
        )
    except Exception as exc:  # database not running or not configured
        return error(503, f"database unavailable: {type(exc).__name__}")
    try:
        with con, con.cursor() as cur:
            cur.execute("CREATE TABLE IF NOT EXISTS visits (id SERIAL PRIMARY KEY, at TIMESTAMP DEFAULT now())")
            cur.execute("INSERT INTO visits DEFAULT VALUES")
            cur.execute("SELECT COUNT(*) FROM visits")
            count = cur.fetchone()[0]
    finally:
        con.close()
    return jsonify(visits=count)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050)
