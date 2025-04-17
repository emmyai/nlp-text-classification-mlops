from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
import os


AML_SUBSCRIPTION_ID = os.environ["AML_SUBSCRIPTION_ID"]
AML_RESOURCE_GROUP = os.environ["AML_RESOURCE_GROUP"]
AML_WORKSPACE = os.environ["AML_WORKSPACE"]
AZURE_CREDENTIALS = os.environ["AZURE_CREDENTIALS"]

ml_client = MLClient(AZURE_CREDENTIALS, AML_SUBSCRIPTION_ID, AML_RESOURCE_GROUP, AML_WORKSPACE)

print("✅ MLflow Tracking URI:", ml_client.workspaces.get(AML_WORKSPACE).mlflow_tracking_uri)
