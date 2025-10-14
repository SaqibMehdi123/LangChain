from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
import os
from langchain_google_genai import GoogleGenerativeAI

load_dotenv()

model = GoogleGenerativeAI(
    model='gemini-1.5-flash',
    temperature=0.5,
    google_api_key = os.getenv("GOOGLE_API_KEY")
)

messages = [
    SystemMessage(content='You are a helpful assistant.'),
    HumanMessage(content='Tell me about LangChain and LangGraph')
]

result = model.invoke(messages)
messages.append(AIMessage(content=result))

print(messages)