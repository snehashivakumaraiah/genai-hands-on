text = """
Python is a programming language.
Python was created by Guido van Rossum.
Python is widely used for web development.
Flask is a Python web framework.
FastAPI is another Python web framework.
"""

chunk_size = 100

chunks = []

for i in range(0,len(text),chunk_size):
    chunk = text[i:i+chunk_size]
    chunks.append(chunk)

print(chunks)
