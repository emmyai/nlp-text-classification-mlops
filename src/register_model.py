import os
import mlflow
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

# Load environment variables for Azure ML
subscription_id = os.environ["AZURE_SUBSCRIPTION_ID"]
resource_group = os.environ["AZURE_RESOURCE_GROUP"]
workspace_name = os.environ["AZURE_WORKSPACE_NAME"]

# Connect to Azure ML workspace
ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id=subscription_id,
    resource_group_name=resource_group,
    workspace_name=workspace_name
)

# Set MLflow tracking URI if needed (optional if already configured)
# mlflow.set_tracking_uri("azureml://...")

# Get the experiment name from your config or hardcode
experiment_name = "NLP-Text-Classification"
client = mlflow.tracking.MlflowClient()

# Get latest run with a logged model
runs = client.search_runs(
    experiment_ids=[client.get_experiment_by_name(experiment_name).experiment_id],
    filter_string="attributes.status = 'FINISHED'",
    order_by=["start_time DESC"],
    max_results=1
)

if not runs:
    raise Exception("No completed MLflow runs found with logged models.")

latest_run = runs[0]
run_id = latest_run.info.run_id
model_uri = f"runs:/{run_id}/model"

# Register the model
registered_model = mlflow.register_model(
    model_uri=model_uri,
    name="nlp-text-classification-model"
)

print(f"Model registered from run {run_id}: {registered_model.name} (v{registered_model.version})")
