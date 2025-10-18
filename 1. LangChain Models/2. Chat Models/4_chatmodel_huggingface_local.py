from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv(filename=".env", usecwd=True))
api_key = os.getenv("GOOGLE_API_KEY")
print(f"GOOGLE_API_KEY loaded: {'YES' if api_key else 'NO'}")

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    model_kwargs={"temperature": 0.5, "max_new_tokens": 512},
)

model = ChatHuggingFace(llm=llm)

messages = [
    ("system", "You are a helpful assistant."),
    ("human", "Explain the chain of thought prompting.")
]
response = model.invoke(messages)
print(response.content)