import pandas as pd

import mlflow

import mlflow.sklearn

from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import classification_report, accuracy_score

import joblib

import os

import json

from azure.identity import ClientSecretCredential

from azure.ai.ml import MLClient

from load_config import load_parameters
 
# Load credentials from environment

azure_credentials = json.loads(os.environ["AZURE_CREDENTIALS"])

subscription_id = os.environ["AML_SUBSCRIPTION_ID"]

resource_group = os.environ["AML_RESOURCE_GROUP"]

workspace_name = os.environ["AML_WORKSPACE"]

tenant_id = os.environ["AZURE_TENANT_ID"]

client_id = os.environ["AZURE_CLIENT_ID"]

client_secret = os.environ["AZURE_CLIENT_SECRET"]
 
# Load configuration

config = load_parameters()

train_config = config["train"]

test_size = train_config["test_size"]

random_state = train_config["random_state"]

max_features = train_config["max_features"]

ngram_range = tuple(train_config["ngram_range"])

model_type = train_config["model_type"]
 
# Initialize ML Client

credential = ClientSecretCredential(tenant_id, client_id, client_secret)

ml_client = MLClient(credential, subscription_id, resource_group, workspace_name)
 
# Set MLflow tracking

tracking_uri = ml_client.workspaces.get(workspace_name).mlflow_tracking_uri

mlflow.set_tracking_uri(tracking_uri)

mlflow.set_experiment("nlp-text-classification")
 
def train():

    # Load and prepare data

    df = pd.read_csv("data/processed/nlp_text_cleaned.csv")

    X = df["text"]

    y = df["label"]

    # Train-test split with stratification

    X_train, X_test, y_train, y_test = train_test_split(

        X, y, 

        test_size=test_size, 

        random_state=random_state, 

        stratify=y

    )
 
    # Base pipeline

    pipeline = Pipeline([

        ("tfidf", TfidfVectorizer(

            max_features=max_features,

            ngram_range=ngram_range,

            stop_words='english',

            min_df=5,       # Ignore rare terms

            max_df=0.95     # Ignore overly common terms

        )),

        ("clf", LogisticRegression())

    ])
 
    # Hyperparameter grid for regularization tuning

    param_grid = {

        'clf__penalty': ['l1', 'l2'],

        'clf__C': [0.001, 0.01, 0.1, 1, 10, 100],

        'clf__solver': ['liblinear', 'saga'],

        'clf__class_weight': [None, 'balanced']

    }
 
    # Grid search with cross-validation

    grid_search = GridSearchCV(

        pipeline,

        param_grid,

        cv=5,

        scoring='accuracy',

        verbose=1,

        n_jobs=-1

    )
 
    with mlflow.start_run() as run:

        # Perform grid search

        grid_search.fit(X_train, y_train)

        # Get best model

        best_model = grid_search.best_estimator_

        # Log best parameters

        mlflow.log_params(grid_search.best_params_)

        mlflow.log_metric("best_cv_score", grid_search.best_score_)

        # Evaluate on test set

        # Save artifacts

        os.makedirs("models/sklearn_model", exist_ok=True)

        joblib.dump(best_model, "models/sklearn_model/best_model.pkl")

        # Save test data

        os.makedirs("data/processed", exist_ok=True)

        pd.DataFrame(X_test).to_csv("data/processed/X_test.csv", index=False)

        pd.DataFrame(y_test).to_csv("data/processed/y_test.csv", index=False)

        # Log model and artifacts

        mlflow.sklearn.log_model(best_model, "model")

        mlflow.log_artifact("data/processed/X_test.csv")

        mlflow.log_artifact("data/processed/y_test.csv")

        print(f"✅ Model trained with regularization. Run ID: {run.info.run_id}")

        print(f"Best parameters: {grid_search.best_params_}")
 
if __name__ == "__main__":

    train()
 