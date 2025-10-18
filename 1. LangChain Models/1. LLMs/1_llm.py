from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv(filename=".env", usecwd=True))
api_key = os.getenv("GOOGLE_API_KEY")
print(f"GOOGLE_API_KEY loaded: {'YES' if api_key else 'NO'}")

llm = GoogleGenerativeAI(
    model="gemini-2.5-flash",   
    temperature=0,
)

messages = [
    ("system", "You are a helpful assistant that answers questions in an interactive way and in just 3 lines."),
    ("human", "What is agentic ai"),
]

ai_message = llm.invoke(messages)

print('\n', ai_message, '\n')
