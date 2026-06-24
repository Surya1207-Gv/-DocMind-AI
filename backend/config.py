import os
from dotenv import load_dotenv

# Directory configurations
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
FAISS_DIR = os.path.join(BASE_DIR, "faiss_indices")

# Load environment variables using absolute path
load_dotenv(os.path.join(BASE_DIR, ".env"))

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Create directories if they don't exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(FAISS_DIR, exist_ok=True)

# Model Settings (Using OpenRouter)
EMBEDDING_MODEL = "openai/text-embedding-3-small"
LLM_MODEL = "nvidia/nemotron-3-nano-30b-a3b:free"

# RAG Settings
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 300
TOP_K = 4
