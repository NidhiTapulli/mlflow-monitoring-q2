import mlflow
import mlflow.sklearn
import joblib

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Iris-Classification")

iris = load_iris()

X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, "model.joblib")

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("random_state", 42)
    mlflow.log_metric("accuracy", accuracy)

    
    mlflow.sklearn.log_model(
        model,
        name="model",
        registered_model_name="IrisClassifier",
        skops_trusted_types=["sklearn.tree._tree.Tree"]
    )
    

    print(f"Model accuracy: {accuracy:.4f}")
    print(f"MLflow run ID: {mlflow.active_run().info.run_id}")

