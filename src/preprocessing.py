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
    df = pd.read_csv("../data/raw/nlp_text_classification_dataset_2000.csv")
    processor = TextPreprocessor()
    df["text"] = processor.transform(df["text"])
    df.to_csv("..data/processed/nlp_text_cleaned.csv", index=False)
    print("Text preprocessing complete and saved to 'nlp_text_cleaned.csv'")