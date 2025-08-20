# gpt_rag_response.py

import os
import openai
import numpy as np
from pinecone import Pinecone

# --- Config / Keys ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "auditopenai"  # your 1024-dim index (text-embedding-3-large with dimensions=1024)
NAMESPACE = ""              # default namespace; change if you segment data

openai.api_key = OPENAI_API_KEY
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# --- Utilities ---

def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalize."""
    v = vec.astype(np.float32)
    norms = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v / norms

def _cosine_sim_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Cosine similarity between rows of A and rows of B."""
    A_n = _l2_normalize(A)
    B_n = _l2_normalize(B)
    return A_n @ B_n.T

def mmr_select(doc_embs: np.ndarray, query_emb: np.ndarray, k: int, lambda_param: float = 0.5):
    """
    Maximal Marginal Relevance:
    Select k indices that are both similar to the query and diverse from one another.
    """
    if doc_embs.size == 0:
        return []

    # Ensure shapes: (num_docs, d), (1, d)
    D = doc_embs.shape[1]
    query = query_emb.reshape(1, D)

    # Similarity to query for each doc
    sim_to_query = _cosine_sim_matrix(doc_embs, query).flatten()

    selected = []
    candidates = list(range(len(doc_embs)))

    # pick the most similar first
    first = int(np.argmax(sim_to_query))
    selected.append(first)
    candidates.remove(first)

    # iteratively add items maximizing MMR
    while len(selected) < min(k, len(doc_embs)) and candidates:
        # similarity of remaining docs to already-selected set
        selected_embs = doc_embs[np.array(selected)]
        sim_to_selected = _cosine_sim_matrix(doc_embs[candidates], selected_embs).max(axis=1)

        # MMR score
        mmr = lambda_param * sim_to_query[candidates] - (1.0 - lambda_param) * sim_to_selected

        # best next candidate
        next_idx_in_candidates = int(np.argmax(mmr))
        next_global_idx = candidates[next_idx_in_candidates]
        selected.append(next_global_idx)
        candidates.remove(next_global_idx)

    return selected

def _embed_query(text: str) -> list:
    """Embed with the same model + dimension as your index."""
    resp = openai.embeddings.create(
        input=text,
        model="text-embedding-3-large",
        dimensions=1024,  # must match index dims
    )
    return resp.data[0].embedding

# --- Retrieval + Context Building ---

def get_context_from_pinecone(
    question: str,
    top_k: int = 10,            # final diverse chunks to return
    candidate_k: int = 50,      # initial wider net from Pinecone before reranking
    per_doc_cap: int = 2,       # max chunks to take from a single Doc_ID
    lambda_param: float = 0.5   # MMR tradeoff (0=diversity, 1=relevance)
) -> str:
    """
    Retrieve candidate neighbors, MMR-rerank for diversity, cap per Doc_ID,
    and return stitched context text.
    """
    # 1) Embed query
    query_vec = _embed_query(question)

    # 2) Retrieve a broader set, include vectors for MMR
    res = index.query(
        vector=query_vec,
        top_k=candidate_k,
        include_metadata=True,
        include_values=True,
        namespace=NAMESPACE
    )

    matches = res.get("matches", []) or []
    if not matches:
        return ""  # no context available

    # 3) Collect embeddings, text, and doc ids
    doc_vectors = []
    doc_texts = []
    doc_ids = []

    for m in matches:
        values = m.get("values")
        meta = m.get("metadata") or {}
        text = meta.get("Content", "")
        doc_id = meta.get("Doc_ID") or meta.get("document_id") or "unknown"

        if values and text:
            doc_vectors.append(values)
            doc_texts.append(text)
            doc_ids.append(doc_id)

    if not doc_vectors:
        return ""

    doc_vectors = np.array(doc_vectors, dtype=np.float32)
    query_vec_np = np.array(query_vec, dtype=np.float32)

    # 4) MMR rerank to choose diverse, relevant candidates
    selected_idxs = mmr_select(doc_vectors, query_vec_np, k=min(top_k * 3, len(doc_vectors)), lambda_param=lambda_param)

    # 5) Enforce per_doc_cap and dedupe text
    per_doc_count = {}
    seen_snippets = set()
    contexts = []

    for idx in selected_idxs:
        did = doc_ids[idx]
        txt = doc_texts[idx].strip()

        # simple text dedupe (prefix hash)
        key = (did, txt[:120])
        if key in seen_snippets:
            continue

        cnt = per_doc_count.get(did, 0)
        if cnt >= per_doc_cap:
            continue

        contexts.append(txt)
        per_doc_count[did] = cnt + 1
        seen_snippets.add(key)

        if len(contexts) >= top_k:
            break

    return "\n\n---\n\n".join(contexts)

# --- GPT Call ---

def ask_gpt(question: str, context: str, model: str = "gpt-4o"):
    system_prompt = (
        "You are an audit intelligence assistant. Users will ask about risk insights, domains, "
        "document types, and specific audit topics. Each content chunk may include fields like "
        "Doc_ID, Domain, Document_Type, and free-form text. "
        "CRITICAL RULES:\n"
        "1) Use ONLY the provided context. If the context is insufficient or does not contain the answer, say: "
        "\"I don’t know based on the provided context.\" Do not guess.\n"
        "2) When asked to aggregate or count, base counts strictly on the chunks in context.\n"
        "3) Prefer concise, structured answers (bullets/tables) and cite Doc_IDs when helpful."
    )

    user_prompt = f"Context:\n{context if context else '[no relevant context returned]'}\n\nQuestion:\n{question}"

    resp = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,  # lower temp to reduce hallucinations further
    )
    return resp.choices[0].message.content.strip()

# --- Public entrypoint used by your FastAPI route ---

def query_audit_gpt(question: str, top_k: int = 10):
    """
    Main entry used by your API: retrieve diverse context then ask GPT.
    """
    context = get_context_from_pinecone(
        question,
        top_k=top_k,
        candidate_k=max(30, top_k * 5),  # widen the initial pool
        per_doc_cap=2,
        lambda_param=0.5,
    )
    return ask_gpt(question, context)
