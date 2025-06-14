import os
import tempfile
import requests
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Qdrant imports
from qdrant_client import QdrantClient, models as qmodels

from text_utils import extract_text, chunk_text, embed_chunks

load_dotenv()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Connect to Qdrant
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
qdrant = QdrantClient(url=QDRANT_URL)

class Query(BaseModel):
    question: str

@app.get("/")
async def root():
    return {"status": "AI-Router is alive!"}

@app.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    # 1. Save file to temp
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        path = tmp.name

    # 2. Extract, chunk, embed
    text = extract_text(path)
    os.remove(path)
    chunks = chunk_text(text)
    embeddings = embed_chunks(chunks)

    # 3. Ensure the 'documents' collection exists
    names = [c.name for c in qdrant.get_collections().collections]
    if "documents" not in names:
        qdrant.create_collection(
            collection_name="documents",
            vector_size=len(embeddings[0]),
            distance=qmodels.Distance.COSINE,
        )

    # 4. Upsert into Qdrant
    points = [
        {"id": idx, "vector": embeddings[idx], "payload": {"text": chunks[idx]}}
        for idx in range(len(chunks))
    ]
    qdrant.upsert(collection_name="documents", points=points)
    return {"status": "ok", "chunks": len(chunks)}

@app.post("/query")
async def query(q: Query):
    embedding = embed_chunks([q.question])[0]

    hits = qdrant.search(
        collection_name="documents",
        query_vector=embedding,
        limit=5,
    )
    if not hits:
        raise HTTPException(404, "No relevant context found")

    context = "\n\n".join(h.payload["text"] for h in hits)
    prompt = f"Context:\n{context}\n\nQuestion: {q.question}"

    resp = requests.post(
        "http://host.docker.internal:11434/v1/chat/completions",
        json={
            "model": "llama2",
            "messages": [
                {"role": "system", "content": "You are The Brain assistant."},
                {"role": "user",   "content": prompt},
            ],
        },
        timeout=30,
    )
    resp.raise_for_status()
    answer = resp.json()["choices"][0]["message"]["content"]
    return {"answer": answer}
