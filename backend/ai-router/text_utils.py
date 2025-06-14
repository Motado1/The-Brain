import os, requests
from PyPDF2 import PdfReader
from docx import Document

def extract_text(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        reader = PdfReader(path)
        return "\\n".join(p.extract_text() or "" for p in reader.pages)
    if ext == ".docx":
        doc = Document(path)
        return "\\n".join(p.text for p in doc.paragraphs)
    with open(path, encoding="utf-8") as f:
        return f.read()

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200):
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def embed_chunks(chunks: list[str]) -> list[list[float]]:
    # calls Ollama embeddings endpoint
    resp = requests.post(
        "http://127.0.0.1:11434/v1/embeddings",
        json={"model":"llama2", "input": chunks}
    )
    resp.raise_for_status()
    data = resp.json()["data"]
    return [d["embedding"] for d in data]