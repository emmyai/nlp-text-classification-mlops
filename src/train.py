import os
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Define paths
DATA_PATH = "data/processed/nlp_text_cleaned.csv"
MODEL_DIR = "models"
X_TEST_PATH = os.path.join(MODEL_DIR, "X_test.pkl")
Y_TEST_PATH = os.path.join(MODEL_DIR, "y_test.pkl")
MODEL_PATH = os.path.join(MODEL_DIR, "text_classification_model.pkl")

# Create model directory if not exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_PATH)
X = df["text"]
y = df["label"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression(max_iter=1000))
])

# MLflow setup
mlflow.set_experiment("NLP-Text-Classification")
with mlflow.start_run():
    pipeline.fit(X_train, y_train)

    # Log model with MLflow
    mlflow.sklearn.log_model(pipeline, "model")

    # Dump model and test data to disk
    joblib.dump(pipeline, MODEL_PATH)
    joblib.dump(X_test, X_TEST_PATH)
    joblib.dump(y_test, Y_TEST_PATH)

    print(f"Model and test sets saved to '{MODEL_DIR}' and logged to MLflow.")
