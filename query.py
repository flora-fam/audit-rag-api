# query.py
import os
import openai
from pinecone import Pinecone

# ✅ Load API keys from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "auditopenai"

# ✅ Initialize OpenAI
openai.api_key = OPENAI_API_KEY

# ✅ Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

def query_pinecone(user_question: str, top_k: int = 5):
    # Step 1: Embed the query using OpenAI
    response = openai.embeddings.create(
        input=user_question,
        model="text-embedding-3-large",
        dimensions=1024
    )
    vector = response.data[0].embedding

    # Step 2: Query Pinecone
    result = index.query(
        vector=vector,
        top_k=top_k,
        include_metadata=True
    )

    # Step 3: Format and return results
    matches = []
    for match in result.get("matches", []):
        matches.append({
            "score": match["score"],
            "id": match["id"],
            "Doc_ID": match["metadata"].get("Doc_ID", ""),
            "Document_Type": match["metadata"].get("Document_Type", ""),
            "Domain": match["metadata"].get("Domain", ""),
            "Last_Updated": match["metadata"].get("Last_Updated", ""),
            "Content": match["metadata"].get("Content", "")
        })

    return
