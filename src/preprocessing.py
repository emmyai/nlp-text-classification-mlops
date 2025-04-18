import os
import pandas as pd
import re
import string
from sklearn.base import BaseEstimator, TransformerMixin

class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.pattern = re.compile(r"[^a-zA-Z\s]")

    def clean_text(self, text):
        text = text.lower()
        text = self.pattern.sub("", text)
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.apply(self.clean_text)

if __name__ == "__main__":
    raw_path = "data/raw/Sentiment_Classification_Dataset.csv"
    output_path = "data/processed/nlp_text_cleaned.csv"
    df = pd.read_csv("data/raw/Sentiment_Classification_Dataset.csv", encoding="ISO-8859-1", errors="replace")

    print(f"📄 Initial dataset shape: {df.shape}")

    # Drop rows with missing text or label
    df.dropna(subset=["text", "label"], inplace=True)
    print(f"🧹 After dropping missing values: {df.shape}")

    # Remove duplicate rows
    df.drop_duplicates(subset=["text", "label"], inplace=True)
    print(f"📌 After removing duplicates: {df.shape}")

    # Apply text cleaning
    processor = TextPreprocessor()
    df["text"] = processor.transform(df["text"])

    # Ensure output directory exists
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"✅ Preprocessing complete. Cleaned data saved to '{output_path}'")
