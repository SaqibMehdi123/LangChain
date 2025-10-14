from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.9
)

messages = [
    ("system", "You are a helpful assistant that answers questions in an interactive way."),
    ("human", "Difference between AI, ML and Deep Learning?"),
]

response = model.invoke(messages)
print('\n', response.content, '\n')