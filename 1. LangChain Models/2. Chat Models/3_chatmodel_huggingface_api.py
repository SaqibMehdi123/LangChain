from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv(filename=".env", usecwd=True))
api_key = os.getenv("GOOGLE_API_KEY")
print(f"GOOGLE_API_KEY loaded: {'YES' if api_key else 'NO'}")

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    max_new_tokens=512,
    temperature=0.5,
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

model = ChatHuggingFace(llm=llm)

messages = [
    ("system", "You are a helpful assistant. Respond concisely."),
    ("human", "Explain the chain of thought prompting.")
]

response = model.invoke(messages)
print(response.content)
