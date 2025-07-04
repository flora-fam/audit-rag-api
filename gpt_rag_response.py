# gpt_rag_response.py

import os
import openai
from pinecone import Pinecone

# Load API keys from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "auditopenai"

openai.api_key = OPENAI_API_KEY
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

def get_context_from_pinecone(question, top_k=5):
    response = openai.embeddings.create(
        input=question,
        model="text-embedding-3-large",
        dimensions=1024
    )
    vector = response.data[0].embedding

    result = index.query(vector=vector, top_k=top_k, include_metadata=True)

    chunks = []
    for match in result.get("matches", []):
        chunks.append(match["metadata"].get("Content", ""))

    return "\n\n---\n\n".join(chunks)

def ask_gpt(question, context, model="gpt-4o"):
    system_prompt = (
    "You are an audit intelligence assistant. Users will ask about risk insights, domains, document types, and specific audit topics. "
    "Each content chunk includes a Doc_ID, Domain, Document_Type, and free-form text. "
    "You should extract relevant information, count document mentions, group by fields when asked, and summarize important risk notes. "
    "Use only the content you’re given — do not guess. Only say 'I don’t know' if the context is completely unrelated."
)


    full_prompt = f"Context:\n{context}\n\nQuestion:\n{question}"

    response = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": full_prompt}
        ],
        temperature=0.3
    )

    return response.choices[0].message.content.strip()

def query_audit_gpt(question, top_k=5):
    context = get_context_from_pinecone(question, top_k)
    return ask_gpt(question, context)
