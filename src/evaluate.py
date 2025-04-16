import joblib
import mlflow
from sklearn.metrics import accuracy_score, classification_report

# Load model and data
model = mlflow.sklearn.load_model("model")
X_test = joblib.load("X_test.pkl")
y_test = joblib.load("y_test.pkl")

# Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)

# Log metrics
mlflow.log_metric("accuracy", accuracy)
for label, metrics in report.items():
    if isinstance(metrics, dict):
        for metric_name, value in metrics.items():
            mlflow.log_metric(f"{label}_{metric_name}", value)

print(f"Evaluation complete. Accuracy: {accuracy:.4f}")