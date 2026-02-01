import faiss
import pickle
import os
import re
import difflib
import logging
from functools import lru_cache
from typing import List, Tuple, Optional
import numpy as np
import requests
from fastembed import TextEmbedding
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("rag_engine")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "mistralai/mistral-7b-instruct")
OPENROUTER_TEMPERATURE = float(os.getenv("OPENROUTER_TEMPERATURE", "0.2"))
OPENROUTER_TIMEOUT = int(os.getenv("OPENROUTER_TIMEOUT", "30"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

INDEX_PATH = "faiss_store/index.faiss"
META_PATH = "faiss_store/meta.pkl"
DEFAULT_TOP_K = int(os.getenv("RAG_TOP_K", "6"))

@lru_cache(maxsize=1)
def _get_embedder() -> TextEmbedding:
    return TextEmbedding(model_name=EMBEDDING_MODEL)

@lru_cache(maxsize=1)
def _get_index():
    return faiss.read_index(INDEX_PATH)

@lru_cache(maxsize=1)
def _get_documents() -> List[str]:
    with open(META_PATH, "rb") as f:
        return pickle.load(f)

def _embed_texts(texts: List[str]) -> np.ndarray:
    embedder = _get_embedder()
    vectors = list(embedder.embed(texts))
    return np.vstack(vectors).astype("float32")

def _keyword_overlap_score(query: str, text: str) -> float:
    q_tokens = set(_tokenize(query))
    t_tokens = set(_tokenize(text))
    if not q_tokens or not t_tokens:
        return 0.0
    return len(q_tokens.intersection(t_tokens)) / max(len(q_tokens), 1)

def retrieve(query: str, top_k: int = DEFAULT_TOP_K) -> List[str]:
    q_embedding = _embed_texts([query])
    index = _get_index()
    documents = _get_documents()
    scores, indices = index.search(q_embedding, max(top_k * 3, top_k))

    candidates = []
    for i, idx in enumerate(indices[0]):
        doc = documents[idx]
        vec_score = float(scores[0][i])
        keyword_score = _keyword_overlap_score(query, doc)
        fuzzy_score = difflib.SequenceMatcher(None, _normalize(query), _normalize(doc)).ratio()
        combined = (vec_score * 0.6) + (keyword_score * 0.25) + (fuzzy_score * 0.15)
        candidates.append((combined, doc))

    candidates.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in candidates[:top_k]]

_STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "is", "are",
    "was", "were", "be", "by", "with", "as", "at", "from", "that", "this",
    "it", "its", "their", "your", "you", "we", "our", "they", "them", "will",
    "should", "can", "may", "must", "only", "not", "if", "then", "than",
}

def _tokenize(text: str) -> List[str]:
    return [t for t in re.findall(r"\b\w+\b", text.lower()) if t not in _STOPWORDS]

def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())

def _is_heading(line: str) -> bool:
    if not line:
        return False
    if re.match(r"^[=\-]{3,}$", line):
        return False
    letters_only = re.sub(r"[^A-Za-z]", "", line)
    if not letters_only:
        return False
    is_all_caps = letters_only.isupper()
    ends_with_colon = line.strip().endswith(":")
    return is_all_caps or ends_with_colon

def _best_match_from_blocks(question: str, blocks: List[str]) -> Tuple[Optional[str], float]:
    lines = []
    for block in blocks:
        for raw in block.splitlines():
            line = raw.strip()
            if line:
                lines.append(line)

    if not lines:
        return None, 0

    q_tokens = set(_tokenize(question))
    if not q_tokens:
        return None, 0

    best_idx = None
    best_score = 0
    q_norm = _normalize(question)
    for i, line in enumerate(lines):
        l_tokens = set(_tokenize(line))
        token_score = len(q_tokens.intersection(l_tokens))
        fuzzy_score = difflib.SequenceMatcher(None, q_norm, _normalize(line)).ratio()
        score = token_score + (fuzzy_score * 2)
        if score > best_score:
            best_score = score
            best_idx = i

    if best_idx is not None and best_score > 0.15:
        line = lines[best_idx]
        if _is_heading(line):
            collected = []
            for follow in lines[best_idx + 1:best_idx + 6]:
                if _is_heading(follow) or re.match(r"^[=\-]{3,}$", follow):
                    break
                collected.append(follow)
            if collected:
                return " ".join(collected), best_score
        return line, best_score

    sentences = []
    for block in blocks:
        for s in re.split(r"(?<=[.!?])\s+", block.strip()):
            if s:
                sentences.append(s.strip())

    if not sentences:
        return None, 0

    best_sentence = None
    best_score = 0
    for sentence in sentences:
        s_tokens = set(_tokenize(sentence))
        token_score = len(q_tokens.intersection(s_tokens))
        fuzzy_score = difflib.SequenceMatcher(None, q_norm, _normalize(sentence)).ratio()
        score = token_score + (fuzzy_score * 2)
        if score > best_score:
            best_score = score
            best_sentence = sentence

    return best_sentence, best_score

def _extract_answer(question: str, context_blocks: List[str], all_blocks: Optional[List[str]] = None) -> str:
    answer, score = _best_match_from_blocks(question, context_blocks)
    if answer and score > 0:
        return answer

    if all_blocks and all_blocks is not context_blocks:
        answer, score = _best_match_from_blocks(question, all_blocks)
        if answer and score > 0:
            return answer

    return "Information not available"

def _build_prompt(question: str, context: str) -> str:
    return f"""
You are a helpful hackathon information assistant.

RULES (must follow exactly):
- Use ONLY the provided CONTEXT to answer.
- If the answer cannot be found in CONTEXT, reply exactly: "Information not available".
- For short/simple factual questions, reply concisely (default ≤ 3 sentences).
- For broad or explicitly requested detailed answers (question contains words like "detail", "explain", "steps", "how to", "comprehensive", "full", "long"), return a structured, longer response with headings and bullet points.
- If the user's input is a simple greeting (e.g., "hi", "hello"), reply exactly: "Hello, this is the PICT InC Assistant. How can I help you?"
- If the user's question has typos or poor grammar, interpret intent and answer as if corrected.
- If the question is unclear and the CONTEXT does not resolve ambiguity, ask one short clarification question.
- Do NOT repeat or mirror the CONTEXT or the QUESTION in the answer.
- Use clear, human-friendly language. Prefer bullets or numbered steps when helpful.
- Always keep the reply focused and relevant to the user's QUESTION.

CONTEXT:
{context}

QUESTION:
{question}
"""

def _is_greeting(text: str) -> bool:
    text = _normalize(text)
    return text in {"hi", "hello", "hallo", "hey", "hi there", "hello there"}

def _clean_output(text: str) -> str:
    cleaned = re.sub(r"[*_`]+", "", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned

def ask_llm(question: str) -> str:
    if not question or not question.strip():
        return "Information not available"

    if _is_greeting(question):
        return "Hello, this is the PICT InC Assistant. How can I help you?"

    context_blocks = retrieve(question)
    context = "\n\n".join(context_blocks)

    if not context.strip():
        return "Information not available"

    prompt = _build_prompt(question, context)

    if not OPENROUTER_API_KEY:
        return _clean_output(_extract_answer(question, context_blocks, _get_documents()))

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": OPENROUTER_MODEL,
                "temperature": OPENROUTER_TEMPERATURE,
                "messages": [
                    {"role": "system", "content": "Answer strictly from context."},
                    {"role": "user", "content": prompt}
                ]
            },
            timeout=OPENROUTER_TIMEOUT
        )
        if response.status_code != 200:
            logger.warning("OpenRouter error status: %s", response.status_code)
            return _clean_output(_extract_answer(question, context_blocks, _get_documents()))

        data = response.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        if not content:
            return _clean_output(_extract_answer(question, context_blocks, _get_documents()))

        if content.strip().lower() == "information not available":
            fallback = _extract_answer(question, context_blocks, _get_documents())
            return _clean_output(fallback)

        return _clean_output(content)
    except requests.RequestException as exc:
        logger.warning("OpenRouter request failed: %s", exc)
        return _clean_output(_extract_answer(question, context_blocks, _get_documents()))
