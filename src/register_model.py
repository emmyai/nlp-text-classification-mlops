import os
import mlflow
from azure.identity import AzureCliCredential
from azure.ai.ml import MLClient

def register_model():
    subscription_id = os.environ["AML_SUBSCRIPTION_ID"]
    resource_group = os.environ["AML_RESOURCE_GROUP"]
    workspace_name = os.environ["AML_WORKSPACE"]

    credential = AzureCliCredential()

    ml_client = MLClient(credential, subscription_id, resource_group, workspace_name)
    tracking_uri = ml_client.workspaces.get(workspace_name).mlflow_tracking_uri

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("NLP-Text-Classification")

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

    registered_model = mlflow.register_model(model_uri=model_uri, name="nlp-text-classification-model")
    print(f"✅ Model registered: {registered_model.name} (v{registered_model.version})")

if __name__ == "__main__":
    register_model()
