from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

llm = GoogleGenerativeAI(
    model="gemini-1.5-flash",   
    temperature=0,
)

messages = [
    ("system", "You are a helpful assistant that answers questions in an interactive way."),
    ("human", "What is agentic ai"),
]

ai_message = llm.invoke(messages)

print('\n', ai_message, '\n')
