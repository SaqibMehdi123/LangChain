from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv(filename=".env", usecwd=True))
api_key = os.getenv("GOOGLE_API_KEY")
print(f"GOOGLE_API_KEY loaded: {'YES' if api_key else 'NO'}")

embeddings = HuggingFaceEndpoint(
	repo_id="Qwen/Qwen3-Embedding-8B",
	task="feature-extraction",
	huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

vector = embeddings.invoke("Hello, world!")
print("Vector length:", len(vector))
print("Vector (first 10 dims):", vector[:10])