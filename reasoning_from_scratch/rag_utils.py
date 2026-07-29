import faiss
import torch
import numpy as np
import json
import os
from transformers import AutoTokenizer, AutoModel

# ✅ Initialize HuggingFace tokenizer + model
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# ✅ File paths for persistence
INDEX_FILE = "faiss_index.bin"
ROWS_FILE = "stored_rows.json"

# ✅ Initialize FAISS index (384 dimensions for MiniLM embeddings)
index = faiss.IndexFlatL2(384)
stored_rows = []

def embed_texts(texts):
    """Convert list of texts into embeddings using mean pooling."""
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.cpu().numpy().astype("float32")

def add_to_index(rows, headers=None):
    """
    Add rows or plain text documents to FAISS index with embeddings.
    - rows: list of dicts or list of strings
    - headers: optional list of keys if rows are dicts
    """
    global stored_rows
    if headers:
        texts = [" ".join(str(r.get(h, "")) for h in headers) for r in rows]
    else:
        # treat rows as plain strings
        texts = [str(r) for r in rows]

    embeddings = embed_texts(texts)
    index.add(embeddings)
    stored_rows.extend(rows)
    save_state()

def query_index(user_query, top_k=5):
    """Retrieve most relevant rows for a query."""
    q_emb = embed_texts([user_query])
    D, I = index.search(q_emb, top_k)
    return [stored_rows[i] for i in I[0] if i < len(stored_rows)]

def save_state():
    """Persist FAISS index and stored rows to disk."""
    faiss.write_index(index, INDEX_FILE)
    with open(ROWS_FILE, "w", encoding="utf-8") as f:
        json.dump(stored_rows, f, ensure_ascii=False, indent=2)

def load_state():
    """Load FAISS index and stored rows from disk if available."""
    global index, stored_rows
    if os.path.exists(INDEX_FILE):
        index = faiss.read_index(INDEX_FILE)
    if os.path.exists(ROWS_FILE):
        with open(ROWS_FILE, "r", encoding="utf-8") as f:
            stored_rows = json.load(f)

# ✅ Load state at startup
load_state()
