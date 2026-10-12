import os
import time
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from flask import Flask, request, jsonify
from prometheus_client import Counter, Histogram, generate_latest
from prometheus_client import CONTENT_TYPE_LATEST
from prometheus_client import make_wsgi_app
from werkzeug.middleware.dispatcher import DispatcherMiddleware

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MODEL_PATH = os.getenv("MODEL_PATH", "model.joblib")

app = Flask(__name__)
PREDICTIONS = Counter("model_predictions_total", "Total model predictions")
REQUEST_TIME = Histogram("prediction_request_seconds", "Prediction request duration")

model = None
try:
    model = joblib.load(MODEL_PATH)
except Exception:
    pass

@app.route("/")
def home():
    return jsonify({
        "application": "Iris ML Model",
        "status": "running",
        "prediction_endpoint": "/predict",
        "metrics_endpoint": "/metrics"
    })

@app.route("/predict", methods=["POST"])
@REQUEST_TIME.time()
def predict():
    global model
    if model is None:
        return jsonify({"error": "Model not loaded. Train the model first."}), 503
    data = request.get_json(silent=True) or {}
    features = data.get("features")
    if not isinstance(features, list) or len(features) != 4:
        return jsonify({"error": "Provide four numeric features."}), 400
    try:
        values = np.array(features, dtype=float).reshape(1, -1)
        result = int(model.predict(values)[0])
        PREDICTIONS.inc()
        return jsonify({"predicted_class": result})
    except (ValueError, TypeError):
        return jsonify({"error": "Features must be numeric."}), 400

@app.route("/metrics")
def metrics():
    from flask import Response
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
