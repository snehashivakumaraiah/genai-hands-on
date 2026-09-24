import os
import requests
import faiss
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


# 1. Load Environment variables

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    raise ValueError("API key is missing")

# 2. Load Document

with open("documents/knowledge.txt","r",encoding = "utf-8") as file:
    text = file.read()

# 3. Split document into chunks

chunk_size = 200
chunks = []

for i in range(0,len(text),chunk_size):
    chunk = text[i:i+chunk_size]
    chunks.append(chunk)

# 4. Creating Embeddings


