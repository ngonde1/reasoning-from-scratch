import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from sentence_transformers import SentenceTransformer, models

# Load transformer backbone
word_embedding_model = models.Transformer(
    "C:/Users/Immanuel/Desktop/reasoning-from-scratch/all-MiniLM-L6-v2"
)

# Add pooling with new API
pooling_model = models.Pooling(
    word_embedding_model.get_embedding_dimension(),
    pooling_mode="mean"
)

# Build SentenceTransformer
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# Encode a test sentence
embedding = model.encode(["hello"])
print("Embedding shape:", embedding.shape)
