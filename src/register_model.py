
import os
from azure.identity import ClientSecretCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model

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

    # Register model directly from local path
    model = Model(
        path="models/model.pkl",
        name="nlp-text-classification-model",
        description="TF-IDF + Logistic Regression model for text classification",
        type="custom_model"
    )

    registered_model = ml_client.models.create_or_update(model)
    print(f"✅ Model registered in Azure ML: {registered_model.name} (v{registered_model.version})")

if __name__ == "__main__":
    register_model()
