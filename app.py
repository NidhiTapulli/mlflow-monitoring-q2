import os
import joblib
import numpy as np
from flask import Flask, request, jsonify, Response
from prometheus_client import Counter, Histogram, generate_latest
from prometheus_client import CONTENT_TYPE_LATEST

MODEL_PATH = os.getenv("MODEL_PATH", "model.joblib")

app = Flask(__name__)

PREDICTIONS = Counter(
    "model_predictions_total",
    "Total model predictions"
)

REQUEST_TIME = Histogram(
    "prediction_request_seconds",
    "Prediction request duration"
)

CLASS_NAMES = ["setosa", "versicolor", "virginica"]

model = None

try:
    model = joblib.load(MODEL_PATH)
except Exception as error:
    print(f"Model could not be loaded: {error}")


@app.route("/")
def home():
    return jsonify({
        "application": "Iris ML Model",
        "status": "running",
        "health_endpoint": "/health",
        "prediction_endpoint": "/predict",
        "metrics_endpoint": "/metrics"
    })


@app.route("/health")
def health():
    model_loaded = model is not None

    return jsonify({
        "status": "healthy" if model_loaded else "unhealthy",
        "model_loaded": model_loaded
    }), 200 if model_loaded else 503


@app.route("/predict", methods=["POST"])
@REQUEST_TIME.time()
def predict():
    if model is None:
        return jsonify({
            "error": "Model not loaded. Train the model first."
        }), 503

    data = request.get_json(silent=True) or {}
    features = data.get("features")

    if not isinstance(features, list) or len(features) != 4:
        return jsonify({
            "error": "Provide four numeric features."
        }), 400

    try:
        values = np.array(features, dtype=float).reshape(1, -1)

        if not np.all(np.isfinite(values)):
            raise ValueError("Features must be finite numbers.")

        result = int(model.predict(values)[0])

        if result < 0 or result >= len(CLASS_NAMES):
            return jsonify({
                "error": "Model returned an unknown class ID."
            }), 500

        PREDICTIONS.inc()

        return jsonify({
            "predicted_class": result,
            "class_name": CLASS_NAMES[result]
        })

    except (ValueError, TypeError):
        return jsonify({
            "error": "Features must be four finite numeric values."
        }), 400


@app.route("/metrics")
def metrics():
    return Response(
        generate_latest(),
        mimetype=CONTENT_TYPE_LATEST
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
