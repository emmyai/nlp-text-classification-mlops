from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
import os

subscription_id = os.environ["AML_SUBSCRIPTION_ID"]
resource_group = os.environ["AML_RESOURCE_GROUP"]
workspace_name = os.environ["AML_WORKSPACE"]
credential = DefaultAzureCredential()

ml_client = MLClient(credential, subscription_id, resource_group, workspace_name)

print("✅ MLflow Tracking URI:", ml_client.workspaces.get(workspace_name).mlflow_tracking_uri)
