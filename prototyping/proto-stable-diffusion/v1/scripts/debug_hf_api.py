import os, requests

# Don't leave tokens in commit-able code. See gitignored tokens.py instead.
HF_TOKEN = None

resp = requests.get(
    "https://api-inference.huggingface.co/models/CompVis/stable-diffusion-v1-4",
    headers={"Authorization": f"Bearer {HF_TOKEN}"}
)
print("Endpoint status:", resp.status_code, resp.reason)
print("Response:", resp.text[:200])

