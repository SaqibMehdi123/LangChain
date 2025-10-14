from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
from langchain_google_genai import GoogleGenerativeAI

load_dotenv()

model = GoogleGenerativeAI(
    model='gemini-1.5-flash',
    temperature=0.5,
    google_api_key = os.getenv("GOOGLE_API_KEY")
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful {domain} expert."),
    ("human", "Explain the concept of {concept} in simple terms.")
])

messages = prompt.format_messages(domain="AI", concept="reinforcement learning")
result = model.invoke(messages)

print(result)