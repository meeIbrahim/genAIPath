from langchain_groq import ChatGroq
import os

key = os.getenv("K")
if not key:
    raise ValueError("C_K environment variable not set")
os.environ["GROQ_API_KEY"] = key.strip()

model = ChatGroq(model="llama3-8b-8192")
messages = [
    (
        "system",
        "You are a helpful translator. Translate the user sentence to French.",
    ),
    (
        "human",
        "I love programming.",
    ),
]
ai_msg = model.invoke(messages)
