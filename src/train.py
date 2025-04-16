import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Load dataset
df = pd.read_csv("/data/raw/nlp_text_classification_dataset_2000.csv")
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
    mlflow.sklearn.log_model(pipeline, "model")
    joblib.dump(X_test, "X_test.pkl")
    joblib.dump(y_test, "y_test.pkl")
    print("Model trained and logged to MLflow.")


