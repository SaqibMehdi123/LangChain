from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()


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

