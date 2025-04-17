import pandas as pd
import joblib
import mlflow
from sklearn.metrics import accuracy_score, classification_report
import os
import json
from azure.identity import ClientSecretCredential
from azure.ai.ml import MLClient

# Load Azure credentials from environment variables
azure_credentials = json.loads(os.environ["AZURE_CREDENTIALS"])
subscription_id = os.environ["AML_SUBSCRIPTION_ID"]
resource_group = os.environ["AML_RESOURCE_GROUP"]
workspace_name = os.environ["AML_WORKSPACE"]
tenant_id = os.environ["AZURE_TENANT_ID"]
client_id = os.environ["AZURE_CLIENT_ID"]
client_secret = os.environ["AZURE_CLIENT_SECRET"]

# Authenticate using ClientSecretCredential
credential = ClientSecretCredential(tenant_id, client_id, client_secret)

# Initialize MLClient and set tracking URI
ml_client = MLClient(credential, subscription_id, resource_group, workspace_name)
tracking_uri = ml_client.workspaces.get(workspace_name).mlflow_tracking_uri
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("nlp-text-classification")

def evaluate():
    # Ensure files exist (model and data saved by train.py)
    model_path = "models/sklearn_model/model.pkl"
    x_test_path = "data/processed/X_test.csv"
    y_test_path = "data/processed/y_test.csv"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at {model_path}. Run train.py first.")
    if not os.path.exists(x_test_path) or not os.path.exists(y_test_path):
        raise FileNotFoundError("Test data not found. Ensure train.py has run successfully.")

    # Load model and test data
    model = joblib.load(model_path)
    X_test = pd.read_csv(x_test_path)
    y_test = pd.read_csv(y_test_path)

    with mlflow.start_run(nested=True):
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        # Log metrics
        mlflow.log_metric("accuracy", acc)
        report = classification_report(y_test, preds)
        mlflow.log_text(report, "classification_report.txt")

        print(f"✅ Evaluation complete. Accuracy: {acc:.4f}")
        print("📋 Classification report logged to MLflow.")

if __name__ == "__main__":
    evaluate()
