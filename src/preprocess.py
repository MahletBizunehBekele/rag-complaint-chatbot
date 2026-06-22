import pandas as pd
import re
from pathlib import Path


RAW_DATA = "data/raw/complaints.csv"
OUTPUT_DATA = "data/processed/filtered_complaints.csv"


TARGET_PRODUCTS = [
    "Credit card",
    "Consumer Loan",
    "Money transfer",
    "Checking or savings account"
]


def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()

    boilerplate_phrases = [
        "i am writing to file a complaint",
        "consumer complaint"
    ]

    for phrase in boilerplate_phrases:
        text = text.replace(phrase, "")

    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def main():

    Path("data/processed").mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RAW_DATA)

    print(f"Original Shape: {df.shape}")

    # EDA 1
    print("\nComplaint Distribution")
    print(df["Product"].value_counts())

    # EDA 2
    with_narrative = df["Consumer complaint narrative"].notna().sum()
    without_narrative = df["Consumer complaint narrative"].isna().sum()

    print(f"\nWith narrative: {with_narrative}")
    print(f"Without narrative: {without_narrative}")

    # EDA 3
    df["word_count"] = (
        df["Consumer complaint narrative"]
        .fillna("")
        .str.split()
        .apply(len)
    )

    print(df["word_count"].describe())

    # Filtering
    filtered_df = df[
        df["Product"].isin(TARGET_PRODUCTS)
    ]

    filtered_df = filtered_df[
        filtered_df["Consumer complaint narrative"].notna()
    ]

    filtered_df = filtered_df[
        filtered_df["Consumer complaint narrative"].str.strip() != ""
    ]

    filtered_df["cleaned_narrative"] = (
        filtered_df["Consumer complaint narrative"]
        .apply(clean_text)
    )

    filtered_df.to_csv(
        OUTPUT_DATA,
        index=False
    )

    print(f"\nSaved: {OUTPUT_DATA}")
    print(filtered_df.shape)


if __name__ == "__main__":
    main()