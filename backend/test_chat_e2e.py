import os
import sys
import requests
import json

def run_test():
    base_url = "http://127.0.0.1:8000"
    
    # 1. Login to get access token
    print("Logging in as 'admin'...")
    login_payload = {
        "username": "admin",
        "password": "admin123"
    }
    response = requests.post(f"{base_url}/api/auth/login", json=login_payload)
    if response.status_code != 200:
        print(f"Login failed: {response.text}")
        return
        
    token_data = response.json()
    token = token_data["access_token"]
    print(f"Logged in successfully. Token: {token[:15]}...")
    
    # 2. Get list of documents
    headers = {
        "Authorization": f"Bearer {token}"
    }
    print("\nFetching documents list...")
    docs_response = requests.get(f"{base_url}/api/documents", headers=headers)
    if docs_response.status_code != 200:
        print(f"Failed to fetch documents: {docs_response.text}")
        return
        
    docs = docs_response.json()
    if not docs:
        print("No documents found in the database. Please make sure AI.pdf is uploaded.")
        return
        
    for d in docs:
        print(f"Doc ID: {d['id']} | Name: {d['name']}")
    
    # Use the first document ID
    doc_id = docs[0]["id"]
    doc_name = docs[0]["name"]
    print(f"Using document: {doc_name} ({doc_id})")
    
    # 3. Call Chat API
    chat_payload = {
        "question": "What is Generative AI?",
        "doc_ids": [doc_id],
        "history": [],
        "mode": "qa"
    }
    
    print(f"\nSending chat query: 'What is Generative AI?'...")
    chat_response = requests.post(f"{base_url}/api/chat", json=chat_payload, headers=headers, stream=True)
    if chat_response.status_code != 200:
        print(f"Chat request failed: {chat_response.text}")
        return
        
    print("\nStreaming response from backend:")
    print("-" * 50)
    
    full_text = ""
    metadata = None
    
    # Parse SSE stream
    for line in chat_response.iter_lines():
        if line:
            line_str = line.decode("utf-8").strip()
            if line_str.startswith("data: "):
                data_content = line_str[6:]
                if data_content == "[DONE]":
                    break
                try:
                    data_json = json.loads(data_content)
                    if data_json.get("type") == "token":
                        token_text = data_json.get("text", "")
                        sys.stdout.write(token_text)
                        sys.stdout.flush()
                        full_text += token_text
                    elif data_json.get("type") == "metadata":
                        metadata = data_json
                except Exception as e:
                    pass
                    
    print("\n" + "-" * 50)
    print("\nFinal Metadata Event:")
    if metadata:
        print(f"Confidence: {metadata.get('confidence')}%")
        print("Sources:")
        for idx, src in enumerate(metadata.get("sources", [])):
            print(f"  [{idx+1}] Page {src.get('page')} | Relevance: {src.get('relevance')}% | Snippet: {src.get('text')[:100].replace('\n', ' ')}...")
        if "content" in metadata:
            print("\nStripped Clean Response Content in final metadata:")
            print(metadata.get("content"))
    else:
        print("No metadata event received!")

if __name__ == "__main__":
    run_test()
