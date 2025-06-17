# app.py
from fastapi import FastAPI
from pydantic import BaseModel
from query import query_pinecone

app = FastAPI()

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5

@app.post("/query-audit-docs")
def query_audit_docs(request: QueryRequest):
    results = query_pinecone(request.question, request.top_k)
    return {"results": results}
