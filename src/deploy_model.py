import os
import uuid
from azure.identity import ClientSecretCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import ManagedOnlineEndpoint, ManagedOnlineDeployment

def deploy_model():
    subscription_id = os.environ["AML_SUBSCRIPTION_ID"]
    resource_group = os.environ["AML_RESOURCE_GROUP"]
    workspace_name = os.environ["AML_WORKSPACE"]
    tenant_id = os.environ["AZURE_TENANT_ID"]
    client_id = os.environ["AZURE_CLIENT_ID"]
    client_secret = os.environ["AZURE_CLIENT_SECRET"]

    credential = ClientSecretCredential(
        tenant_id=tenant_id,
        client_id=client_id,
        client_secret=client_secret
    )

    ml_client = MLClient(credential, subscription_id, resource_group, workspace_name)

    # Fetch latest model version dynamically
    model_name = "nlp-text-classification-model"
    registered_models = ml_client.models.list(name=model_name)
    latest_model = max(registered_models, key=lambda x: int(x.version))
    latest_model_version = latest_model.version
    print(f"✅ Latest model version: {latest_model_version}")

    endpoint_name = f"nlp-endpoint-{uuid.uuid4().hex[:6]}"

    endpoint = ManagedOnlineEndpoint(
        name=endpoint_name,
        auth_mode="key"
    )
    print(f"🚀 Creating endpoint: {endpoint_name}")
    ml_client.begin_create_or_update(endpoint).result()

    deployment = ManagedOnlineDeployment(
        name="default",
        endpoint_name=endpoint_name,
        model=f"{model_name}:{latest_model_version}",
        instance_type="Standard_DS3_v2",  # Updated SKU
        instance_count=1
    )
    print(f"🚀 Deploying model version {latest_model_version} to endpoint: {endpoint_name}")
    ml_client.begin_create_or_update(deployment).result()

    ml_client.online_endpoints.begin_update(
        endpoint_name=endpoint_name,
        default_deployment_name="default"
    ).result()

    print(f"✅ Deployment successful at endpoint: {endpoint_name}")

if __name__ == "__main__":
    deploy_model()
