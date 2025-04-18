import pandas as pd
import joblib
import mlflow
import numpy as np
from sklearn.metrics import (accuracy_score, classification_report, 
                            precision_recall_fscore_support, confusion_matrix)
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
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
 
# Load configuration
config = load_parameters()
train_config = config["train"]
evaluate_config = config["evaluate"]
 
# Configuration parameters
max_features = train_config["max_features"]
ngram_range = tuple(train_config["ngram_range"])
model_path = evaluate_config["model_path"]
x_test_path = evaluate_config["x_test_path"]
y_test_path = evaluate_config["y_test_path"]
full_data_path = evaluate_config["full_data_path"]
cv_folds = evaluate_config.get("cv_folds", 5)
 
# Authenticate and initialize MLClient
credential = ClientSecretCredential(tenant_id, client_id, client_secret)
ml_client = MLClient(credential, subscription_id, resource_group, workspace_name)
tracking_uri = ml_client.workspaces.get(workspace_name).mlflow_tracking_uri
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("nlp-text-classification")
 
def log_confusion_matrix(y_true, y_pred, class_names):
    """Generate and log confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    # Save and log to MLflow
    temp_path = "confusion_matrix.png"
    plt.savefig(temp_path)
    mlflow.log_artifact(temp_path)
    plt.close()
    os.remove(temp_path)
 
def evaluate():
    # Validate paths
    required_files = [model_path, x_test_path, y_test_path, full_data_path]
    for file_path in required_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Required file not found: {file_path}")
 
    # Load model and data
    trained_model = joblib.load(model_path)
    X_test = pd.read_csv(x_test_path)["text"]
    y_test = pd.read_csv(y_test_path)["label"]
    full_df = pd.read_csv(full_data_path)
    X_full = full_df["text"]
    y_full = full_df["label"]
    # Get class names from the trained model
    class_names = trained_model.classes_.tolist()
 
    with mlflow.start_run(nested=True):
        # Standard test set evaluation
        y_pred = trained_model.predict(X_test)
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted'
        )
        # Log metrics
        mlflow.log_metrics({
            "test_accuracy": accuracy,
            "test_precision": precision,
            "test_recall": recall,
            "test_f1": f1
        })
        # Generate and log classification report
        report = classification_report(y_test, y_pred, target_names=class_names)
        mlflow.log_text(report, "classification_report.txt")
        # Log confusion matrix
        log_confusion_matrix(y_test, y_pred, class_names)
        print(f"✅ Test set evaluation complete. Accuracy: {accuracy:.4f}")
        print("📋 Classification report and confusion matrix logged.")
        # Cross-validation with fresh pipeline (matching training configuration)
        print(f"🔁 Running {cv_folds}-fold cross-validation...")
        # Create pipeline matching the training configuration
        cv_pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                stop_words='english',
                min_df=5,
                max_df=0.95
            )),
            ("clf", LogisticRegression(
                penalty=trained_model.named_steps['clf'].penalty,
                C=trained_model.named_steps['clf'].C,
                solver=trained_model.named_steps['clf'].solver,
                class_weight=trained_model.named_steps['clf'].class_weight,
                max_iter=1000,
                random_state=train_config["random_state"]
            ))
        ])
        # Run cross-validation
        cv_scores = cross_val_score(
            cv_pipeline, 
            X_full, 
            y_full, 
            cv=cv_folds, 
            scoring="accuracy",
            n_jobs=-1
        )
        # Log CV results
        mlflow.log_metrics({
            "cv_mean_accuracy": np.mean(cv_scores),
            "cv_std_accuracy": np.std(cv_scores)
        })
        # Log individual fold scores
        for i, score in enumerate(cv_scores):
            mlflow.log_metric(f"cv_fold_{i}_accuracy", score)
        print(f"✅ Cross-validation complete. Mean accuracy: {np.mean(cv_scores):.4f} (±{np.std(cv_scores):.4f})")
        print("Individual fold scores:", [f"{s:.4f}" for s in cv_scores])
 
if __name__ == "__main__":
    evaluate()