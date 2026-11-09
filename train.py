"""Train the digit classifier once and save it with its metadata.

    python train.py [output_dir]      # default: model/

The Docker build runs this, so every container serves the same model instead of
retraining at start-up.
"""
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

SEED = 42
TEST_SIZE = 0.2


def train(out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    digits = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(
        digits.data, digits.target, test_size=TEST_SIZE, random_state=SEED, stratify=digits.target
    )
    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)
    accuracy = float(model.score(X_test, y_test))

    joblib.dump(model, out / "model.joblib")
    np.savez_compressed(out / "test_set.npz", X=X_test, y=y_test)
    metadata = {
        "model": "LogisticRegression(max_iter=2000)",
        "dataset": "scikit-learn load_digits (8x8 images, 10 classes)",
        "n_features": int(digits.data.shape[1]),
        "pixel_range": [0, 16],
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "test_accuracy": round(accuracy, 4),
        "seed": SEED,
        "data_sha256": hashlib.sha256(digits.data.tobytes() + digits.target.tobytes()).hexdigest(),
        "scikit_learn": sklearn.__version__,
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


if __name__ == "__main__":
    meta = train(sys.argv[1] if len(sys.argv) > 1 else "model")
    print(f"Saved model: test accuracy {meta['test_accuracy']:.4f} on {meta['n_test']} images")
