from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv(filename=".env", usecwd=True))
api_key = os.getenv("GOOGLE_API_KEY")
print(f"GOOGLE_API_KEY loaded: {'YES' if api_key else 'NO'}")

model = ChatAnthropic(
    model="claude-2",
    temperature=0.9
)

messages = [
    ("system", "You are a helpful assistant."),
    ("human", "Explain the concept of chain-of-thought prompting.")
]

response = model.invoke(messages)
print(response)

