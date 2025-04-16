import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("nlp-text-classification")

def train():
    df = pd.read_csv("data/processed/nlp_text_cleaned.csv")
    X = df["text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LogisticRegression(max_iter=500))
    ])

    with mlflow.start_run() as run:
        pipeline.fit(X_train, y_train)

        # ✅ Ensure 'models/' folder exists
        # ✅ After (save in a directory):
        os.makedirs("models/sklearn_model", exist_ok=True)
        joblib.dump(pipeline, "models/sklearn_model/model.pkl")
        # Save model
        joblib.dump(pipeline, "models/model.joblib")
        mlflow.sklearn.log_model(pipeline, "model")
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("test_size", 0.2)

        # Save test set for evaluation
        os.makedirs("data/processed", exist_ok=True)
        X_test.to_csv("data/processed/X_test.csv", index=False)
        y_test.to_csv("data/processed/y_test.csv", index=False)

        print(f"✅ Model trained and logged. Run ID: {run.info.run_id}")

if __name__ == "__main__":
    train()
