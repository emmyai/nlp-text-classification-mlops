from azure.ai.ml import MLClient
from azure.identity import ClientSecretCredential
from azure.ai.ml.entities import Model
import os

def register_model():
    # Use ClientSecretCredential for GitHub Actions authentication
    credential = ClientSecretCredential(
        tenant_id=os.getenv("AZURE_TENANT_ID"),
        client_id=os.getenv("AZURE_CLIENT_ID"),
        client_secret=os.getenv("AZURE_CLIENT_SECRET")
    )

    ml_client = MLClient(
        credential=credential,
        subscription_id=os.getenv("AZURE_SUBSCRIPTION_ID"),
        resource_group=os.getenv("AZURE_RESOURCE_GROUP"),
        workspace_name=os.getenv("AZURE_WORKSPACE_NAME"),
    )

    model = Model(
        path="models/model.joblib",
        name="nlp-text-classification-model",
        description="LogReg + TF-IDF NLP Classifier",
        type="custom_model"
    )

    ml_client.models.create_or_update(model)
    print("✅ Model registered in Azure ML.")

if __name__ == "__main__":
    register_model()
