from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "Python is easy to learn.",
    "Python is simple for beginners.",
    "I like eating pizza."
]

embeddings = model.encode(sentences)

for sentence, embedding in zip(sentences, embeddings):
    print("\nSentence:", sentence)
    print("Vector length:", len(embedding))