import requests

url = "https://openrouter.ai/api/v1/models"
response = requests.get(url)
response.raise_for_status()
data = response.json()

free_models = []
for model in data["data"]:
    pricing = model.get("pricing", {})
    # Check if prompt and completion pricing is zero
    prompt_price = float(pricing.get("prompt", 0))
    completion_price = float(pricing.get("completion", 0))
    
    if prompt_price == 0.0 and completion_price == 0.0:
        free_models.append(model["id"])

print("--- Active Free Models on OpenRouter ---")
for m in sorted(free_models):
    print(m)
