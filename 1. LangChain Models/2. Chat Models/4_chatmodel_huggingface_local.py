from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv
import os

load_dotenv()

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