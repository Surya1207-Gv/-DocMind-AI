import os
import google.generativeai as genai
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load absolute path of .env
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

api_key = os.getenv("GOOGLE_API_KEY")
print(f"API Key starting with: {api_key[:6] if api_key else 'None'}... (length: {len(api_key) if api_key else 0})")

# Initialize genai SDK
genai.configure(api_key=api_key)

print("\n--- Listing available models for this key ---")
try:
    models = genai.list_models()
    for m in models:
        print(f"Model: {m.name} | Supported methods: {m.supported_generation_methods}")
except Exception as e:
    print(f"Error listing models: {str(e)}")

models_to_test = [
    "models/text-embedding-004",
    "text-embedding-004",
    "models/embedding-001",
    "embedding-001",
]

for model in models_to_test:
    print(f"\nTesting model: {model}...")
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model=model,
            google_api_key=api_key
        )
        vector = embeddings.embed_query("Hello world test query")
        print(f"[SUCCESS] Vector dimension: {len(vector)}")
        print(f"SUGGESTED_MODEL_NAME = '{model}'")
        break
    except Exception as e:
        print(f"[FAILED] Error: {str(e)}")
