import faiss
import pickle
import re
from sentence_transformers import SentenceTransformer
from pathlib import Path

MODEL_NAME = "all-MiniLM-L6-v2"
INDEX_PATH = "faiss_store/index.faiss"
META_PATH = "faiss_store/meta.pkl"

model = SentenceTransformer(MODEL_NAME)

text = Path("data/hackathon.txt").read_text(encoding="utf-8")

def split_sentences(paragraph: str):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", paragraph) if s.strip()]

def chunk_text(text, target_size=700, overlap_sentences=1):
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    sentences = []
    for para in paragraphs:
        sentences.extend(split_sentences(para))

    chunks = []
    current = []
    current_len = 0
    for sentence in sentences:
        if current_len + len(sentence) + 1 > target_size and current:
            chunks.append(" ".join(current))
            overlap = current[-overlap_sentences:] if overlap_sentences > 0 else []
            current = overlap[:]
            current_len = sum(len(s) for s in current) + max(len(current) - 1, 0)

        current.append(sentence)
        current_len += len(sentence) + 1

    if current:
        chunks.append(" ".join(current))

    return chunks

chunks = chunk_text(text)

embeddings = model.encode(chunks, normalize_embeddings=True)

dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

faiss.write_index(index, INDEX_PATH)

with open(META_PATH, "wb") as f:
    pickle.dump(chunks, f)

print(f"✅ FAISS index built with {len(chunks)} chunks")
