import pandas as pd
import sys

if __name__ == "__main__":
    try:
        df = pd.read_csv("..data/raw/nlp_text_classification_dataset_2000.csv")

        # Basic checks
        assert not df.empty, "Dataset is empty."
        assert 'text' in df.columns and 'label' in df.columns, "Missing required columns."
        assert df['text'].isnull().sum() == 0, "Text column contains null values."
        assert df['label'].isnull().sum() == 0, "Label column contains null values."
        assert df['label'].nunique() >= 2, "Label column must contain at least 2 classes."

        print("Data validation passed ✅")
    except Exception as e:
        print(f"Data validation failed ❌: {e}")
        sys.exit(1)