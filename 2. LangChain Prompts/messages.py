from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv, find_dotenv
import os
from langchain_google_genai import GoogleGenerativeAI

load_dotenv(find_dotenv(filename=".env", usecwd=True))
api_key = os.getenv("GOOGLE_API_KEY")
print(f"GOOGLE_API_KEY loaded: {'YES' if api_key else 'NO'}")

model = GoogleGenerativeAI(
    model='gemini-2.5-flash',
    temperature=0.5,
    google_api_key=api_key
)

messages = [
    SystemMessage(content='You are a helpful assistant. Provide concise and informative answers.'),
    HumanMessage(content='Tell me about LangChain and LangGraph')
]

result = model.invoke(messages)
messages.append(AIMessage(content=result))

print(messages)