from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(
    base_url = "https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENAI_API_KEY")
)

response = client.chat.completions.create(
    model = "nvidia/nemotron-3-ultra-550b-a55b:free",
    messages = [
           {
        "role": "system",
        "content": "You are a senior Python developer. Explain concepts clearly using simple examples."
    },
    {
        "role": "user",
        "content": "Explain decorators in Python."
    }
    ]
)
print(response.choices[0].message.content)