import os
import mlflow
from azure.identity import ClientSecretCredential
from azure.ai.ml import MLClient
import register_model  # Azure MLflow plugin

def register_model_pipeline():
    subscription_id = os.environ["AML_SUBSCRIPTION_ID"]
    resource_group = os.environ["AML_RESOURCE_GROUP"]
    workspace_name = os.environ["AML_WORKSPACE"]

    credential = ClientSecretCredential(
        tenant_id=os.environ["AZURE_TENANT_ID"],
        client_id=os.environ["AZURE_CLIENT_ID"],
        client_secret=os.environ["AZURE_CLIENT_SECRET"]
    )

    ml_client = MLClient(credential, subscription_id, resource_group, workspace_name)

    # Set MLflow to track to Azure
    tracking_uri = f"azureml://subscriptions/{subscription_id}/resourceGroups/{resource_group}/workspaces/{workspace_name}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("NLP-Text-Classification")

    # Get latest run
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("NLP-Text-Classification")
    latest_run = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["start_time DESC"],
        max_results=1
    )[0]

    run_id = latest_run.info.run_id
    model_uri = f"runs:/{run_id}/model"

    # Use AzureML plugin to register the model
    registered_model = register_model(
        model_uri=model_uri,
        workspace=ml_client,
        model_name="nlp-text-classification-model"
    )

    print(f"✅ Model registered in Azure ML: {registered_model.name}")

if __name__ == "__main__":
    register_model_pipeline()
