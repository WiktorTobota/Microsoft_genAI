from pathlib import Path
from dotenv import load_dotenv
import os

# Load env vars
default_env = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=default_env)

# Paths
dir_root = Path(__file__).parent
PERSIST_DIR = dir_root / 'Outputs' / 'Part2_outputs' / 'chroma-persist'
MODEL_DIR = dir_root / 'Outputs' / 'Part2_outputs' / 'Sentence transformer'
PERSIST_DIR.mkdir(parents=True, exist_ok=True)

# ChromaDB and embeddings
from sentence_transformers import SentenceTransformer
from Tools.EmbeddingFunction import EmbeddingFunction
from chromadb import PersistentClient
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE

BASE_MODEL_NAME = 'all-MiniLM-L6-v2'
COLLECTION_NAME = 'Baza1'

# Load sentence transformer model
if MODEL_DIR.exists() and any(MODEL_DIR.iterdir()):
    base_model = SentenceTransformer(str(MODEL_DIR), device='cpu')
else:
    base_model = SentenceTransformer(BASE_MODEL_NAME, device='cpu')

# Create embedding function
embed_fn = EmbeddingFunction(base_model)

# Initialize Chroma client and collection
client = PersistentClient(
    path=str(PERSIST_DIR),
    tenant=DEFAULT_TENANT,
    database=DEFAULT_DATABASE
)
collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=embed_fn
)
