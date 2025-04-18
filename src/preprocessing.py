import os
import pandas as pd
import re
import string
from sklearn.base import BaseEstimator, TransformerMixin

class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.pattern = re.compile(r"[^a-zA-Z\s]")

    def clean_text(self, text):
        if pd.isnull(text):
            return ""
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
    
    # ✅ Read with encoding fix
    try:
        df = pd.read_csv(raw_path, encoding="ISO-8859-1")
    except Exception as e:
        print(f"❌ Failed to read file: {e}")
        exit(1)

    # ✅ Drop missing values
    df.dropna(subset=["text", "label"], inplace=True)

    # ✅ Drop duplicates
    df.drop_duplicates(inplace=True)

    # ✅ Clean text
    processor = TextPreprocessor()
    df["text"] = processor.transform(df["text"])

    # ✅ Ensure output directory exists
    os.makedirs("data/processed", exist_ok=True)

    # ✅ Save cleaned dataset
    df.to_csv("data/processed/nlp_text_cleaned.csv", index=False, encoding="utf-8")
    print("✅ Text preprocessing complete and saved to 'nlp_text_cleaned.csv'")
