import os
import requests
import faiss
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


# 1. Load Environment variables

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")

print("API key loaded:", API_KEY is not None)
print("API key length:", len(API_KEY) if API_KEY else 0)

if not API_KEY:
    raise ValueError("API key is missing")

# 2. Load Document

with open("knowledge.txt","r",encoding = "utf-8") as file:
    text = file.read()

# 3. Split document into chunks

chunk_size = 200
chunks = []

for i in range(0,len(text),chunk_size):
    chunk = text[i:i+chunk_size]
    chunks.append(chunk)

# 4. Creating Embeddings

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks)
print("Embeddings shape:",embeddings.shape)

# 5. Create FAISS Index

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
print("Vectors stored in faiss:",index.ntotal)

# 6. Retrive relevant chunks

def retrieve(question, k=2):

    question_embedding = model.encode([question])
    distances, indices = index.search(
        question_embedding,
        k
    )
    retrieved_chunks = []

    for i in indices[0]:
        retrieved_chunks.append(chunks[i])

    return retrieved_chunks

# 7. Ask the LLM

def generate_answer(question, context):

    prompt = f"""
You are a helpful assistant.

Answer the question using ONLY the context provided below.

Context:
{context}

Question:
{question}

If the answer is not present in the context, say:
"I don't know based on the provided information."
"""

    headers = {
        "Authorization": "Bearer " + API_KEY.strip(),
        "Content-Type": "application/json"
    }

    print("Authorization header created:", headers["Authorization"].startswith("Bearer "))
    print("API key used:", API_KEY[:10] + "..." + API_KEY[-4:])

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json={
            "model": "nvidia/nemotron-3-ultra-550b-a55b:free",
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }
    )

    print("Status:", response.status_code)

    if response.status_code != 200:
        print("Response:", response.text)

    response.raise_for_status()

    result = response.json()

    return result["choices"][0]["message"]["content"]

# 8. RAG Pipeline

question = input("Ask a question:")
retrieved_chunks = retrieve(question)
context = "\n\n".join(retrieved_chunks)
print ("\nRetrieved Context:")
print(context)

answer = generate_answer(question,context)
print("\nAnswer:")
print(answer)
