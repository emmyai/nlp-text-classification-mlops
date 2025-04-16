import pandas as pd
import joblib
import mlflow
from sklearn.metrics import accuracy_score, classification_report
import os

# Set MLflow experiment and tracking URI
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("nlp-text-classification")

def evaluate():
    # Ensure files exist
    if not os.path.exists("models/model.joblib"):
        raise FileNotFoundError("Trained model not found. Run train.py first.")
    if not os.path.exists("data/processed/X_test.csv") or not os.path.exists("data/processed/y_test.csv"):
        raise FileNotFoundError("Test data not found. Ensure train.py has run successfully.")

    # Load model and test data
    model = joblib.load("models/model.joblib")
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_test = pd.read_csv("data/processed/y_test.csv")

    with mlflow.start_run(nested=True):
        preds = model.predict(X_test["text"])
        acc = accuracy_score(y_test, preds)

        # Log metrics
        mlflow.log_metric("accuracy", acc)
        report = classification_report(y_test, preds)
        mlflow.log_text(report, "classification_report.txt")

        print(f"✅ Evaluation complete. Accuracy: {acc:.4f}")
        print("📋 Classification report logged to MLflow.")

if __name__ == "__main__":
    evaluate()
