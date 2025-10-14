from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    max_new_tokens=512,
    temperature=0.5,
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

model = ChatHuggingFace(llm=llm)

messages = [
    ("system", "You are a helpful assistant."),
    ("human", "Explain the chain of thought prompting.")
]

response = model.invoke(messages)
print(response.content)
