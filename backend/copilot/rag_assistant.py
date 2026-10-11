
from pathlib import Path

import faiss
import numpy as np
import ollama
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# Project folders
BASE_DIR = Path(__file__).resolve().parent.parent
DOCUMENTS_DIR = BASE_DIR / "documents"

# AI models
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OLLAMA_MODEL = "llama3.2"

_model = None
_index = None
_chunks = []


def get_embedding_model():
    global _model

    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)

    return _model


def split_text(text, chunk_size=700, overlap=100):
    """Split text into overlapping chunks."""
    text = " ".join(text.split())
    chunks = []

    step = chunk_size - overlap

    if step <= 0:
        raise ValueError("Chunk size must be greater than overlap.")

    for start in range(0, len(text), step):
        chunk = text[start:start + chunk_size].strip()

        if chunk:
            chunks.append(chunk)

    return chunks


def load_documents():
    """Read PDF and TXT files from the documents folder."""
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    all_chunks = []

    for file_path in DOCUMENTS_DIR.iterdir():
        try:
            if file_path.suffix.lower() == ".pdf":
                reader = PdfReader(str(file_path))

                for page in reader.pages:
                    text = page.extract_text() or ""
                    all_chunks.extend(split_text(text))

            elif file_path.suffix.lower() == ".txt":
                text = file_path.read_text(encoding="utf-8-sig")
                all_chunks.extend(split_text(text))

        except Exception as error:
            print(f"Could not read {file_path.name}: {error}")

    return all_chunks


def build_index():
    """Create a searchable FAISS index from document embeddings."""
    global _index, _chunks

    _chunks = load_documents()

    if not _chunks:
        _index = None
        return 0

    model = get_embedding_model()

    embeddings = model.encode(
        _chunks,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    embeddings = np.asarray(embeddings, dtype="float32")

    _index = faiss.IndexFlatIP(embeddings.shape[1])
    _index.add(embeddings)

    return len(_chunks)


def answer_question(question, top_k=4):
    """Retrieve relevant document passages and ask Ollama."""
    if _index is None or not _chunks:
        return {
            "answer": (
                "No documents have been indexed. "
                "Add a PDF or TXT file and build the index first."
            ),
            "sources": []
        }

    model = get_embedding_model()

    query_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )

    k = min(top_k, len(_chunks))
    scores, indices = _index.search(query_embedding, k)

    retrieved_chunks = []
    sources = []

    for score, chunk_index in zip(scores[0], indices[0]):
        if chunk_index < 0:
            continue

        retrieved_chunks.append(_chunks[chunk_index])

        sources.append({
            "chunk": int(chunk_index + 1),
            "similarity": round(float(score), 3)
        })

    context = "\n\n".join(retrieved_chunks)

    prompt = f"""
You are an AI document assistant.

Answer using only the supplied document context.
If the answer is not available in the context, say so.
Do not invent facts.

DOCUMENT CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )

    return {
        "answer": response["message"]["content"],
        "sources": sources
    }
