import pandas as pd
import joblib
import mlflow
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import json
from azure.identity import ClientSecretCredential
from azure.ai.ml import MLClient
from load_config import load_parameters

# Load Azure credentials from environment variables
azure_credentials = json.loads(os.environ["AZURE_CREDENTIALS"])
subscription_id = os.environ["AML_SUBSCRIPTION_ID"]
resource_group = os.environ["AML_RESOURCE_GROUP"]
workspace_name = os.environ["AML_WORKSPACE"]
tenant_id = os.environ["AZURE_TENANT_ID"]
client_id = os.environ["AZURE_CLIENT_ID"]
client_secret = os.environ["AZURE_CLIENT_SECRET"]



config = load_parameters()
train_config = config["train"]


max_features = train_config["max_features"]
ngram_range = tuple(train_config["ngram_range"])

evaluate_config = config["evaluate"]

model_path = evaluate_config["model_path"]
x_test_path = evaluate_config["x_test_path"]
y_test_path = evaluate_config["y_test_path"]
full_data_path = evaluate_config["full_data_path"]

# Authenticate using ClientSecretCredential
credential = ClientSecretCredential(tenant_id, client_id, client_secret)

# Initialize MLClient and set tracking URI
ml_client = MLClient(credential, subscription_id, resource_group, workspace_name)
tracking_uri = ml_client.workspaces.get(workspace_name).mlflow_tracking_uri
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("nlp-text-classification")

def evaluate():

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at {model_path}. Run train.py first.")
    if not os.path.exists(x_test_path) or not os.path.exists(y_test_path):
        raise FileNotFoundError("Test data not found. Ensure train.py has run successfully.")
    if not os.path.exists(full_data_path):
        raise FileNotFoundError("Full training data not found for cross-validation.")

    # Load model and test data
    trained_model = joblib.load(model_path)
    X_test = pd.read_csv(x_test_path)
    y_test = pd.read_csv(y_test_path)

    # Load full dataset for CV
    full_df = pd.read_csv(full_data_path)
    X_full = full_df["text"]
    y_full = full_df["label"]

    with mlflow.start_run(nested=True):
        # Standard test set evaluation
        preds = trained_model.predict(X_test["text"])
        acc = accuracy_score(y_test, preds)

        mlflow.log_metric("test_accuracy", acc)
        report = classification_report(y_test, preds)
        mlflow.log_text(report, "classification_report.txt")

        print(f"\u2705 Test set accuracy: {acc:.4f}")   
        print("\ud83d\udccb Test classification report logged.")

        # Cross-validation score with fresh pipeline to avoid leakage
        print("\ud83d\udd01 Running cross-validation (cv=5)...")
        pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, stop_words='english')),
        ("clf", LogisticRegression())  # adjust based on model_type if needed
        ])
        cv_scores = cross_val_score(pipeline, X_full, y_full, cv=5, scoring="accuracy")
        mlflow.log_metric("cv_mean_accuracy", cv_scores.mean())
        print("\u2705 Cross-validation complete.")
        print("CV Scores:", cv_scores)
        print("Mean CV Accuracy:", cv_scores.mean())

if __name__ == "__main__":
    evaluate()
