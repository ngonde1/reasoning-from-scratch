import faiss
import numpy as np

def main():
    # ✅ Step 1: Create some dummy embeddings (pretend vectors)
    # Let's say we have 4 items, each with 3‑dimensional embeddings
    embeddings = np.array([
        [0.1, 0.2, 0.3],
        [0.2, 0.1, 0.0],
        [0.9, 0.8, 0.7],
        [0.4, 0.5, 0.6]
    ], dtype="float32")

    print("Embeddings shape:", embeddings.shape)

    # ✅ Step 2: Build FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # ✅ Step 3: Run a similarity search
    query = np.array([[0.1, 0.2, 0.25]], dtype="float32")  # a query vector
    D, I = index.search(query, k=2)

    print("\nQuery vector:", query)
    print("Distances:", D)
    print("Indices:", I)

    # ✅ Step 4: Show which items were closest
    for idx in I[0]:
        print("Closest item index:", idx, "embedding:", embeddings[idx])

if __name__ == "__main__":
    main()
