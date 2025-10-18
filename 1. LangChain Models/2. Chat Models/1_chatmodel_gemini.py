from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv(filename=".env", usecwd=True))
api_key = os.getenv("GOOGLE_API_KEY")
print(f"GOOGLE_API_KEY loaded: {'YES' if api_key else 'NO'}")

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.9
)

messages = [
    ("system", "You are a helpful assistant that answers questions in an interactive way and in just 3 lines."),
    ("human", "Difference between AI, ML and Deep Learning?"),
]

response = model.invoke(messages)
print('\n', response.content, '\n')