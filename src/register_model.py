import os
from azure.identity import ClientSecretCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from datetime import datetime

def register_model():
    subscription_id = os.environ["AML_SUBSCRIPTION_ID"]
    resource_group = os.environ["AML_RESOURCE_GROUP"]
    workspace_name = os.environ["AML_WORKSPACE"]

    credential = ClientSecretCredential(
        tenant_id=os.environ["AZURE_TENANT_ID"],
        client_id=os.environ["AZURE_CLIENT_ID"],
        client_secret=os.environ["AZURE_CLIENT_SECRET"]
    )


    ml_client = MLClient(credential, subscription_id, resource_group, workspace_name)

    # Generate safe unique model name
    timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S')
    model_name = f"nlp-text-model-{timestamp}"

    model = Model(
        path="models/sklearn_model",
        name=model_name,
        description="Logistic Regression text classifier",
        type="custom_model"
    )

    registered_model = ml_client.models.create_or_update(model)
    print(f"✅ Model registered: {registered_model.name}, version: {registered_model.version}")

if __name__ == "__main__":
    register_model()
