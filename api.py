from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from rag_engine import ask_llm

BASE_DIR = Path(__file__).resolve().parent
UI_DIR = BASE_DIR / "ui"
STATIC_DIR = UI_DIR / "static"

app = FastAPI(
    title="FAISS RAG Hackathon Bot",
    description="RAG assistant for hackathon information with FAISS + Sentence Transformers",
    version="1.0.0",
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

class Query(BaseModel):
    question: str = Field(..., min_length=2, max_length=1000)

class Answer(BaseModel):
    answer: str

@app.get("/")
def root():
    return FileResponse(UI_DIR / "index.html")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ask", response_model=Answer)
def ask(query: Query):
    try:
        answer = ask_llm(query.question)
        return {"answer": answer}
    except Exception:
        return JSONResponse(status_code=500, content={"answer": "Information not available"})
