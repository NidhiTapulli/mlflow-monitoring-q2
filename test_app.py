import numpy as np
import app as iris_app


class FakeModel:
    def predict(self, values):
        assert values.shape == (1, 4)
        return np.array([0])


def setup_function():
    iris_app.app.config["TESTING"] = True
    iris_app.model = FakeModel()


def test_home_endpoint():
    client = iris_app.app.test_client()
    response = client.get("/")
    assert response.status_code == 200
    assert response.json["status"] == "running"
    assert response.json["prediction_endpoint"] == "/predict"


def test_valid_prediction():
    client = iris_app.app.test_client()
    response = client.post(
        "/predict",
        json={"features": [5.1, 3.5, 1.4, 0.2]}
    )
    assert response.status_code == 200
    assert response.json["predicted_class"] == 0


def test_prediction_rejects_wrong_number_of_features():
    client = iris_app.app.test_client()
    response = client.post(
        "/predict",
        json={"features": [5.1, 3.5]}
    )
    assert response.status_code == 400
    assert "error" in response.json


def test_prediction_rejects_non_numeric_features():
    client = iris_app.app.test_client()
    response = client.post(
        "/predict",
        json={"features": ["abc", 3.5, 1.4, 0.2]}
    )
    assert response.status_code == 400
    assert "error" in response.json


def test_metrics_endpoint():
    client = iris_app.app.test_client()
    response = client.get("/metrics")
    assert response.status_code == 200
    assert b"model_predictions_total" in response.data
