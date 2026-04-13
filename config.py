from dotenv import load_dotenv
import os

load_dotenv()

# Neo4j
NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

# Embedding model (sentence-transformers)
EMBEDDING_MODEL     = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM       = 384          # all-MiniLM-L6-v2 output dim

# FAISS index path
FAISS_INDEX_PATH    = "data/faiss_query.index"
QUERY_META_PATH     = "data/query_meta.json"

# Retrieval hyper-parameters
TOP_K_SIMILAR       = 15           # FAISS top-k candidates
SIMILARITY_THRESHOLD = 0.45        # cosine similarity cutoff
MAX_GRAPH_HOPS      = 2            # BFS hops on schema graph

# Routing score weights  (α, β, γ) — must sum to 1
ALPHA = 0.45   # embedding similarity
BETA  = 0.35   # access count weight
GAMMA = 0.20   # community coverage

# Community detection
COMMUNITY_RESOLUTION = 1.0         # Leiden resolution parameter
MIN_COMMUNITY_SIZE   = 3           # prune tiny communities

# Online learning
DECAY_FACTOR        = 0.95         # exponential decay for old accesses
MIN_ACCESS_COUNT    = 1            # minimum edge weight to keep
