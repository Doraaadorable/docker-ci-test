import json
import sys
from pathlib import Path

import pytest
from flask.testing import FlaskClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app as service  # noqa: E402
import train  # noqa: E402


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("model")
    train.train(out)
    return out


@pytest.fixture
def client(model_dir: Path, monkeypatch: pytest.MonkeyPatch) -> FlaskClient:
    monkeypatch.setattr(service, "MODEL_DIR", model_dir)
    service.artifacts.cache_clear()
    return service.app.test_client()


def test_training_is_reproducible(model_dir: Path, tmp_path: Path) -> None:
    again = train.train(tmp_path)
    first = json.loads((model_dir / "metadata.json").read_text())
    assert again == first
    assert first["test_accuracy"] > 0.9


def test_home_and_health(client: FlaskClient) -> None:
    assert "Test accuracy" in client.get("/").get_data(as_text=True)
    body = client.get("/health").get_json()
    assert body["status"] == "ok" and body["test_accuracy"] > 0.9 and body["pod"]


def test_metadata(client: FlaskClient) -> None:
    body = client.get("/metadata").get_json()
    assert body["n_features"] == 64 and body["pixel_range"] == [0, 16]
    assert len(body["data_sha256"]) == 64


def test_sample_bounds(client: FlaskClient) -> None:
    n_test = client.get("/metadata").get_json()["n_test"]
    assert client.get("/sample/0").status_code == 200
    assert client.get(f"/sample/{n_test - 1}").status_code == 200
    response = client.get(f"/sample/{n_test}")
    assert response.status_code == 404 and "error" in response.get_json()


def test_predict_matches_label_on_held_out_samples(client: FlaskClient) -> None:
    correct = 0
    for i in range(50):
        sample = client.get(f"/sample/{i}").get_json()
        body = client.post("/predict", json={"pixels": sample["pixels"]}).get_json()
        assert 0 <= body["prediction"] <= 9 and 0 < body["probability"] <= 1
        correct += body["prediction"] == sample["label"]
    assert correct >= 45


@pytest.mark.parametrize(
    "payload",
    [
        {},  # missing pixels
        {"pixels": [0] * 3},  # wrong length
        {"pixels": "0" * 64},  # not a list
        {"pixels": ["1"] * 64},  # strings
        {"pixels": [True] * 64},  # booleans
        {"pixels": [0] * 63 + [17]},  # above range
        {"pixels": [0] * 63 + [-1]},  # below range
        [0] * 64,  # JSON array, not an object
    ],
)
def test_predict_rejects_bad_input(client: FlaskClient, payload: object) -> None:
    response = client.post("/predict", json=payload)
    assert response.status_code == 400
    assert "error" in response.get_json()


def test_predict_rejects_non_json(client: FlaskClient) -> None:
    response = client.post("/predict", data="pixels", content_type="text/plain")
    assert response.status_code == 400


def test_unknown_route_returns_json_404(client: FlaskClient) -> None:
    response = client.get("/nope")
    assert response.status_code == 404 and "error" in response.get_json()


def test_visits_without_database_returns_503(client: FlaskClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DB_HOST", "127.0.0.1")
    monkeypatch.setenv("DB_NAME", "does_not_exist")
    response = client.get("/visits")
    assert response.status_code == 503 and "database unavailable" in response.get_json()["error"]
