import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier 
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os
import json
from azure.identity import ClientSecretCredential
from azure.ai.ml import MLClient
from load_config import load_parameters


# Load credentials from the AZURE_CREDENTIALS secret
azure_credentials = json.loads(os.environ["AZURE_CREDENTIALS"])

subscription_id = os.environ["AML_SUBSCRIPTION_ID"]
resource_group = os.environ["AML_RESOURCE_GROUP"]
workspace_name = os.environ["AML_WORKSPACE"]
tenant_id=os.environ["AZURE_TENANT_ID"]
client_id=os.environ["AZURE_CLIENT_ID"]
client_secret=os.environ["AZURE_CLIENT_SECRET"]

from load_config import load_parameters

config = load_parameters()
train_config = config["train"]

test_size = train_config["test_size"]
random_state = train_config["random_state"]
max_features = train_config["max_features"]
ngram_range = tuple(train_config["ngram_range"])
model_type = train_config["model_type"]


# Authenticate using ClientSecretCredential
credential = ClientSecretCredential(tenant_id, client_id, client_secret)

# Initialize MLClient
ml_client = MLClient(credential, subscription_id, resource_group, workspace_name)

# Print MLflow tracking URI
tracking_uri = ml_client.workspaces.get(workspace_name).mlflow_tracking_uri
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("nlp-text-classification")

def train():
    df = pd.read_csv("data/processed/nlp_text_cleaned.csv")
    X = df["text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, stop_words='english')),
    ("clf", LogisticRegression())  # adjust based on model_type if needed
    ])

    with mlflow.start_run() as run:
        pipeline.fit(X_train, y_train)

        # Ensure model directory exists
        os.makedirs("models/sklearn_model", exist_ok=True)

        # Save the model artifact locally
        joblib.dump(pipeline, "models/sklearn_model/model_1.pkl")

        # Log model to Azure ML via MLflow
        mlflow.sklearn.log_model(pipeline, artifact_path="model")

        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("test_size", 0.25)

        # Save test set for evaluation
        os.makedirs("data/processed", exist_ok=True)
        X_test.to_csv("data/processed/X_test.csv", index=False)
        y_test.to_csv("data/processed/y_test.csv", index=False)

        print(f"✅ Model trained and logged. Run ID: {run.info.run_id}")

if __name__ == "__main__":
    train()
