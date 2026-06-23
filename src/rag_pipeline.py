import faiss
import pickle
import numpy as np

from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Load embedding model
embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

INDEX_PATH = BASE_DIR / "vector_store" / "complaints.faiss"
META_PATH = BASE_DIR / "vector_store" / "metadata.pkl"

index = faiss.read_index(str(INDEX_PATH))

with open(META_PATH, "rb") as f:
    metadata = pickle.load(f)


from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained(
    "google/flan-t5-base"
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    "google/flan-t5-base"
)

PROMPT_TEMPLATE = """
You are a financial analyst assistant for CrediTrust.

Use ONLY the complaint excerpts provided below.

If the answer cannot be determined from the context,
say:
"I do not have enough information from the retrieved complaints."

Context:
{context}

Question:
{question}

Answer:
"""


def retrieve(question, k=5):

    query_embedding = embedding_model.encode(
        [question]
    ).astype(np.float32)

    distances, indices = index.search(
        query_embedding,
        k
    )

    retrieved_chunks = []

    for idx in indices[0]:
        if idx < len(metadata):
            retrieved_chunks.append(
                metadata[idx]["chunk_text"]
            )

    return retrieved_chunks

def ask(question):

    chunks = retrieve(question)

    context = "\n\n".join(chunks)

    prompt = PROMPT_TEMPLATE.format(
        context=context,
        question=question
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=150
    )

    answer = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )
    return answer, chunks