from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv, find_dotenv
import os
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv(find_dotenv(filename=".env", usecwd=True))
api_key = os.getenv("GOOGLE_API_KEY")
print(f"GOOGLE_API_KEY loaded: {'YES' if api_key else 'NO'}")

embedding = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    dimensions=300
)

document = [
    "Artificial Intelligence is rapidly transforming industries by enabling machines to perform tasks that previously required human intelligence and decision-making.",
    "Machine Learning provides systems with the ability to automatically learn and improve from experience without being explicitly programmed for every task.",
    "Deep Learning, a subset of machine learning, relies on layered neural networks to analyze large amounts of data and recognize complex patterns with high accuracy.",
    "Natural Language Processing is widely used in chatbots, translation services, and search engines to allow machines to understand, interpret, and generate human language.",
    "Reinforcement Learning focuses on training intelligent agents through trial and error, rewarding correct actions and penalizing incorrect ones to optimize long-term performance."
]

query = 'Tell me about Artificial Intelligence'

doc_vectors = embedding.embed_documents(document)
query_vector = embedding.embed_query(query)

similarities = cosine_similarity([query_vector], doc_vectors).flatten()

print("\nSimilarity scores:", similarities, '\n')
most_similar_idx = np.argmax(similarities)
print("Most similar document:", document[most_similar_idx], '\n')
print("Similarity score:", similarities[most_similar_idx])
