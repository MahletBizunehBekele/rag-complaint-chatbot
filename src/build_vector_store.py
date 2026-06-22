import pandas as pd
import faiss
import numpy as np
import pickle

from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter


INPUT_FILE = "data/processed/filtered_complaints.csv"

VECTOR_DIR = "vector_store"


def create_sample(df):

    sample_size = 12000

    sample_df = (
        df.groupby("Product", group_keys=False)
        .apply(
            lambda x: x.sample(
                frac=sample_size / len(df),
                random_state=42
            )
        )
    )

    return sample_df


def create_chunks(df):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    texts = []
    metadata = []

    for _, row in tqdm(df.iterrows(), total=len(df)):

        chunks = splitter.split_text(
            row["cleaned_narrative"]
        )

        for idx, chunk in enumerate(chunks):

            texts.append(chunk)

            metadata.append(
                {
                    "complaint_id": row["Complaint ID"],
                    "product_category": row["Product"],
                    "chunk_index": idx
                }
            )

    return texts, metadata


def main():

    df = pd.read_csv(INPUT_FILE)

    sample_df = create_sample(df)

    print(sample_df["Product"].value_counts())

    texts, metadata = create_chunks(sample_df)

    print(f"Chunks: {len(texts)}")

    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    embeddings = model.encode(
        texts,
        show_progress_bar=True
    )

    embeddings = np.array(
        embeddings,
        dtype=np.float32
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    faiss.write_index(
        index,
        f"{VECTOR_DIR}/complaints.faiss"
    )

    with open(
        f"{VECTOR_DIR}/metadata.pkl",
        "wb"
    ) as f:
        pickle.dump(metadata, f)

    print("Vector store saved.")


if __name__ == "__main__":
    main()