import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier 
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os
import json
from azure.identity import ClientSecretCredential
from azure.ai.ml import MLClient

# Load credentials from the AZURE_CREDENTIALS secret
azure_credentials = json.loads(os.environ["AZURE_CREDENTIALS"])

subscription_id = os.environ["AML_SUBSCRIPTION_ID"]
resource_group = os.environ["AML_RESOURCE_GROUP"]
workspace_name = os.environ["AML_WORKSPACE"]
tenant_id=os.environ["AZURE_TENANT_ID"]
client_id=os.environ["AZURE_CLIENT_ID"]
client_secret=os.environ["AZURE_CLIENT_SECRET"]

# Authenticate using ClientSecretCredential
credential = ClientSecretCredential(tenant_id, client_id, client_secret)

# Initialize MLClient
ml_client = MLClient(credential, subscription_id, resource_group, workspace_name)

# Print MLflow tracking URI
tracking_uri = ml_client.workspaces.get(workspace_name).mlflow_tracking_uri
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("nlp-text-classification_01")

def train():
    df = pd.read_csv("data/processed/nlp_text_cleaned.csv")
    X = df["text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", DecisionTreeClassifier())
    ])

    with mlflow.start_run() as run:
        pipeline.fit(X_train, y_train)

        # Ensure model directory exists
        os.makedirs("models/sklearn_model", exist_ok=True)

        # Save the model artifact locally
        joblib.dump(pipeline, "models/sklearn_model/model.pkl")

        # Log model to Azure ML via MLflow
        mlflow.sklearn.log_model(pipeline, artifact_path="model")

        mlflow.log_param("model_type", "DecisionTreeClassifier")
        mlflow.log_param("test_size", 0.25)

        # Save test set for evaluation
        os.makedirs("data/processed", exist_ok=True)
        X_test.to_csv("data/processed/X_test.csv", index=False)
        y_test.to_csv("data/processed/y_test.csv", index=False)

        print(f"✅ Model trained and logged. Run ID: {run.info.run_id}")

if __name__ == "__main__":
    train()
