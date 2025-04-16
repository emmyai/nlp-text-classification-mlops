import mlflow
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Model

# Connect to Azure ML workspace
ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id="<your-subscription-id>",
    resource_group_name="<your-resource-group>",
    workspace_name="<your-workspace-name>"
)

# Register model from MLflow
run_id = "<your-run-id>"  # Replace this or automate fetching latest
model_uri = f"runs:/{run_id}/model"

registered_model = mlflow.register_model(
    model_uri=model_uri,
    name="nlp-text-classification-model"
)

print(f"Model registered: {registered_model.name}, version: {registered_model.version}")