from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)
messages = [
    "What is artificial intelligence?",
    "Explain the concept of machine learning.",
    "How does deep learning differ from traditional algorithms?"
]
vectors = embeddings.embed_documents(messages)
for i, vector in enumerate(vectors, 1):
    print(f"Message {i} (first 10 dims):", vector[:10])