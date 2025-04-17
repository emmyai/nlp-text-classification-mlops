import os
import json
from azure.identity import ClientSecretCredential
from azure.ai.ml import MLClient
import mlflow

def register_model():
    # Load Azure environment variables
    subscription_id = os.environ["AML_SUBSCRIPTION_ID"]
    resource_group = os.environ["AML_RESOURCE_GROUP"]
    workspace_name = os.environ["AML_WORKSPACE"]
    tenant_id = os.environ["AZURE_TENANT_ID"]
    client_id = os.environ["AZURE_CLIENT_ID"]
    client_secret = os.environ["AZURE_CLIENT_SECRET"]

    # Authenticate securely with Azure
    credential = ClientSecretCredential(
        tenant_id=tenant_id,
        client_id=client_id,
        client_secret=client_secret
    )

    # Initialize MLClient
    ml_client = MLClient(credential, subscription_id, resource_group, workspace_name)

    # Set MLflow tracking URI to Azure ML
    tracking_uri = ml_client.workspaces.get(workspace_name).mlflow_tracking_uri
    mlflow.set_tracking_uri(tracking_uri)

    experiment_name = "nlp-text-classification"

    # Get the latest run from MLflow experiment
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"MLflow experiment '{experiment_name}' not found.")

    runs = mlflow.search_runs(experiment_ids=experiment.experiment_id, order_by=["start_time desc"], max_results=1)
    if runs.empty:
        raise ValueError(f"No MLflow runs found in experiment '{experiment_name}'.")

    latest_run_id = runs.iloc[0]["run_id"]
    model_uri = f"runs:/{latest_run_id}/model"

    print(f"✅ Registering model from MLflow run ID: {latest_run_id}")

    # Register MLflow model to Azure ML directly
    model_name = "decision-tree-model_01"
    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name=model_name
    )

    print(f"✅ Model successfully registered: '{registered_model.name}', version: '{registered_model.version}'")

if __name__ == "__main__":
    register_model()
