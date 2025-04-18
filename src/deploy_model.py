import os
import uuid
from azure.identity import ClientSecretCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import ManagedOnlineEndpoint, ManagedOnlineDeployment

def deploy_model():
    # Load Azure credentials from environment
    subscription_id = os.environ["AML_SUBSCRIPTION_ID"]
    resource_group = os.environ["AML_RESOURCE_GROUP"]
    workspace_name = os.environ["AML_WORKSPACE"]
    tenant_id = os.environ["AZURE_TENANT_ID"]
    client_id = os.environ["AZURE_CLIENT_ID"]
    client_secret = os.environ["AZURE_CLIENT_SECRET"]

    # Authenticate
    credential = ClientSecretCredential(tenant_id, client_id, client_secret)
    ml_client = MLClient(credential, subscription_id, resource_group, workspace_name)

    # Define endpoint name (you can also persist this if needed)
    endpoint_name = "nlp-text-endpoint"
    print(f"Check: endpoint {endpoint_name} exists")

    # Create endpoint if it doesn't exist
    try:
        ml_client.online_endpoints.get(endpoint_name)
        print(f"✅ Endpoint '{endpoint_name}' already exists.")
    except Exception:
        print(f"🚀 Creating endpoint: {endpoint_name}")
        endpoint = ManagedOnlineEndpoint(
            name=endpoint_name,
            auth_mode="key"
        )
        ml_client.begin_create_or_update(endpoint).result()

    # Get the latest version of the model
    model_name = "Logistic_Regression_model"
    registered_models = ml_client.models.list(name=model_name)
    latest_model = max(registered_models, key=lambda m: int(m.version))
    latest_version = latest_model.version
    print(f"✅ Latest model version: {latest_version}")

    # Deploy model to endpoint
    deployment = ManagedOnlineDeployment(
        name="default",
        endpoint_name=endpoint_name,
        model=f"{model_name}:{latest_version}",
        instance_type="Standard_F2s_v2",  # You can adjust based on quota
        instance_count=1
    )
    print(f"🚀 Deploying model version {latest_version} to endpoint: {endpoint_name}")
    ml_client.begin_create_or_update(deployment).result()

    # Set deployment as default
    print("🔁 Setting deployment as default...")
    endpoint = ml_client.online_endpoints.get(endpoint_name)
    endpoint.defaults = {"deployment_name": "default"}
    ml_client.begin_create_or_update(endpoint).result()

    print(f"✅ Deployment successful. Endpoint: {endpoint_name}, Model version: {latest_version}")

if __name__ == "__main__":
    deploy_model()
