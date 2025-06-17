from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = SentenceTransformer("paraphrase-MiniLM-L3-v2")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index("audit-docs-free")

@app.get("/query")
def query_pinecone(question: str, namespace: str = "default"):
    vector = model.encode([question])[0].tolist()
    result = index.query(
        vector=vector,
        top_k=3,
        include_metadata=True,
        namespace=namespace
    )
    return {"matches": result["matches"]}
