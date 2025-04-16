from azure.ai.ml import MLClient
from azure.ai.ml.entities import ManagedOnlineEndpoint, ManagedOnlineDeployment
from azure.identity import DefaultAzureCredential
import uuid

# Azure ML client
ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id="<your-subscription-id>",
    resource_group_name="<your-resource-group>",
    workspace_name="<your-workspace-name>"
)

# Unique endpoint name
endpoint_name = f"nlp-endpoint-{uuid.uuid4().hex[:6]}"

# Create endpoint
endpoint = ManagedOnlineEndpoint(
    name=endpoint_name,
    auth_mode="key"
)
ml_client.begin_create_or_update(endpoint).result()

# Create deployment
deployment = ManagedOnlineDeployment(
    name="default",
    endpoint_name=endpoint_name,
    model="nlp-text-classification-model:1",
    instance_type="Standard_DS2_v2",
    instance_count=1
)
ml_client.begin_create_or_update(deployment).result()

# Set as default
ml_client.online_endpoints.begin_update(
    endpoint_name=endpoint_name,
    default_deployment_name="default"
).result()

print(f"Model deployed to endpoint: {endpoint_name}")