import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


# Load embedding model
# -------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")

# Knowledge base documents
documents = [
    "Fragile packages must be handled with reduced gripper pressure and not stacked.",
    "Heavy packages require reinforced lifting equipment.",
    "Standard packages can be handled using normal conveyor flow.",
    "Damaged items should be isolated and inspected immediately.",
    "Temperature-sensitive goods must be stored in climate-controlled zones.",
    "Hazardous materials require specialized protective gear and ventilation.",
    "Liquid containers must be kept upright to prevent spillage.",
    "Electronics should be shielded from static and moisture.",
    "Automated Guided Vehicles (AGVs) must yield to human personnel.",
    "Emergency stops must be tested daily before operations begin.",
    "Flammable items must be stored in fire-resistant cabinets.",
    "Oversized loads require a two-person lift or mechanical assistance."
]

# Generate embeddings
doc_embeddings = model.encode(documents)
doc_embeddings = np.array(doc_embeddings).astype("float32")

# Create FAISS index
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(doc_embeddings)


# -------------------------
# Query Function
# -------------------------

def query_knowledge_base(query_text):
    query_embedding = model.encode([query_text])
    query_embedding = np.array(query_embedding).astype("float32")

    D, I = index.search(query_embedding, k=1)
    return documents[I[0][0]]