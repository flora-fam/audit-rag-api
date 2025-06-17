# app.py

from fastapi import FastAPI
from pydantic import BaseModel
from query import query_pinecone
from gpt_rag_response import query_audit_gpt  # ✅ GPT integration

app = FastAPI()

# ✅ Input schema
class QueryRequest(BaseModel):
    question: str
    top_k: int = 5

# ✅ Route 1: Pinecone search only
@app.post("/query-audit-docs")
def query_audit_docs(request: QueryRequest):
    results = query_pinecone(request.question, request.top_k)
    return {"results": results}

# ✅ Route 2: GPT + Pinecone context
@app.post("/ask-audit-gpt")
def ask_audit_gpt(request: QueryRequest):
    answer = query_audit_gpt(request.question, request.top_k)
    return {"answer": answer}
