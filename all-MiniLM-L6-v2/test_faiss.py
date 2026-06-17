import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from sentence_transformers import SentenceTransformer, models
import faiss
import numpy as np

# Step 1: Build SentenceTransformer manually
word_embedding_model = models.Transformer(
    "C:/Users/Immanuel/Desktop/reasoning-from-scratch/all-MiniLM-L6-v2"
)
pooling_model = models.Pooling(
    word_embedding_model.get_embedding_dimension(),
    pooling_mode="mean"
)
st_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# Step 2: Create FAISS index
sentences = [
    "The cat sat on the mat.",
    "Dogs are loyal animals.",
    "Artificial intelligence is transforming the world.",
    "I love eating pizza on weekends."
]

# Encode sentences
embeddings = st_model.encode(sentences)

# Build FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# Step 3: Run a similarity search
query = "Tell me about pets"
query_vec = st_model.encode([query])

D, I = index.search(np.array(query_vec), k=2)

print("\nQuery:", query)
print("Top matches:")
for idx in I[0]:
    print("-", sentences[idx])
