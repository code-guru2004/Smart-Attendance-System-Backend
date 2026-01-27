import pickle

EMBEDDINGS_FILE = "embeddings.pkl"

with open(EMBEDDINGS_FILE, "rb") as f:
    EMBEDDINGS = pickle.load(f)

print(f"Loaded {len(EMBEDDINGS)} face embeddings into memory")
