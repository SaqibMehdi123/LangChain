# from langchain_openai import OpenAIEmbeddings
# from dotenv import load_dotenv

# load_dotenv()

# embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=32)

# result = embedding.embed_query("Delhi is the capital of India")

# print(str(result))

# ----------------------------------------------------------

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(
	model="models/gemini-embedding-001",
	google_api_key=os.getenv("GOOGLE_API_KEY")
)
query = "Islamabad is the capital of Pakistan."
vector = embeddings.embed_query(query)
print("Vector length:", len(vector))
print("Vector (first 10 dims):", vector[:10])