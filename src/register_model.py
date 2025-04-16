import os
from azure.identity import ClientSecretCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from datetime import datetime

def register_model():
    # Load Azure environment variables
    subscription_id = os.environ["AML_SUBSCRIPTION_ID"]
    resource_group = os.environ["AML_RESOURCE_GROUP"]
    workspace_name = os.environ["AML_WORKSPACE"]
    tenant_id=os.environ["AZURE_TENANT_ID"],
    client_id=os.environ["AZURE_CLIENT_ID"],
    client_secret=os.environ["AZURE_CLIENT_SECRET"]

    # Authenticate securely with Azure
    credential = ClientSecretCredential(
        tenant_id=tenant_id,
        client_id=client_id,
        client_secret=client_secret
    )

    # Initialize MLClient
    ml_client = MLClient(credential, subscription_id, resource_group, workspace_name)

    # Define a unique model name using timestamp
    timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S')
    model_name = f"nlp-text-classification-model-{timestamp}"

    # Register the model from local folder
    model = Model(
        path="models/sklearn_model",
        name=model_name,
        description="Logistic Regression model for NLP text classification.",
        type="custom_model",
        tags={
            "framework": "sklearn",
            "task": "text-classification"
        }
    )

    # Register (upload) the model to Azure ML
    registered_model = ml_client.models.create_or_update(model)

    print(f"✅ Model successfully registered: '{registered_model.name}', version: '{registered_model.version}'")

if __name__ == "__main__":
    register_model()
