import os
import requests
from dotenv import load_dotenv
from typing import List, Optional, Any
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

api_key = os.getenv("OPENROUTER_API_KEY")
print(f"OpenRouter Key: {api_key[:12]}... (length: {len(api_key) if api_key else 0})")

class OpenRouterEmbeddings(Embeddings):
    def __init__(self, model: str, api_key: str):
        self.model = model
        self.api_key = api_key
        self.url = "https://openrouter.ai/api/v1/embeddings"
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "input": texts
        }
        response = requests.post(self.url, headers=headers, json=payload)
        if response.status_code != 200:
            print(f"Embedding error: {response.text}")
        response.raise_for_status()
        data = response.json()
        return [item["embedding"] for item in data["data"]]
        
    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

class OpenRouterChat(BaseChatModel):
    model: str
    api_key: str
    temperature: float = 0.2
    
    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs: Any) -> ChatResult:
        formatted_messages = []
        for msg in messages:
            role = "user"
            if msg.type == "assistant" or msg.type == "ai":
                role = "assistant"
            elif msg.type == "system":
                role = "system"
            formatted_messages.append({"role": role, "content": msg.content})
            
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": formatted_messages,
            "temperature": self.temperature
        }
        
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload
        )
        if response.status_code != 200:
            print(f"Chat error response text: {response.text}")
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])

    @property
    def _llm_type(self) -> str:
        return "openrouter-chat"

# 1. Test Embeddings
print("\n--- Testing OpenRouter Embeddings ---")
try:
    embeddings = OpenRouterEmbeddings(
        model="openai/text-embedding-3-small",
        api_key=api_key
    )
    vector = embeddings.embed_query("Hello world test query")
    print(f"[SUCCESS] Embedding dimensions: {len(vector)}")
except Exception as e:
    print(f"[FAILED] Embeddings: {str(e)}")

# 2. Test LLM Chat (Free Models Loop)
print("\n--- Testing OpenRouter LLM Chat ---")
models_to_test = [
    "meta-llama/llama-3.2-3b-instruct:free",
    "google/gemma-4-26b-a4b-it:free",
    "liquid/lfm-2.5-1.2b-instruct:free",
    "google/gemma-4-31b-it:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "cognitivecomputations/dolphin-mistral-24b-venice-edition:free",
]

for model in models_to_test:
    print(f"Testing model: {model}...")
    try:
        llm = OpenRouterChat(
            model=model,
            api_key=api_key
        )
        response = llm.invoke("Hello, say 'OpenRouter working' and nothing else.")
        print(f"[SUCCESS] Chat response: {response.content}")
        print(f"SUGGESTED_FREE_LLM_MODEL = '{model}'")
        break
    except Exception as e:
        print(f"[FAILED] Chat: {str(e)}\n")
