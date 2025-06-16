import requests

API_KEY = "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"
url = "https://openrouter.ai/api/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

data = {
    "model": "meta-llama/llama-3.3-8b-instruct:free",  # or another model listed on openrouter.ai
    "messages": [
        {"role": "user", "content": "What is the capital of France?"}
    ]
}

response = requests.post(url, headers=headers, json=data)
print(response.json())
