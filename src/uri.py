import os
import json
from azure.identity import ClientSecretCredential
from azure.ai.ml import MLClient

# Load credentials from the AZURE_CREDENTIALS secret
azure_credentials = json.loads(os.environ["AZURE_CREDENTIALS"])

subscription_id = os.environ["AML_SUBSCRIPTION_ID"]
resource_group = os.environ["AML_RESOURCE_GROUP"]
workspace_name = os.environ["AML_WORKSPACE"]
tenant_id=os.environ["AZURE_TENANT_ID"]
client_id=os.environ["AZURE_CLIENT_ID"]
client_secret=os.environ["AZURE_CLIENT_SECRET"]

# Authenticate using ClientSecretCredential
credential = ClientSecretCredential(tenant_id, client_id, client_secret)

# Initialize MLClient
ml_client = MLClient(credential, subscription_id, resource_group, workspace_name)

# Print MLflow tracking URI
tracking_uri = ml_client.workspaces.get(workspace_name).mlflow_tracking_uri
print(f"✅ MLflow Tracking URI: {tracking_uri}")

