from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

model = GoogleGenerativeAI(
    model='gemini-1.5-flash',
    temperature=0.5,
    google_api_key = os.getenv("GOOGLE_API_KEY")
)

# chat template
chat_template = ChatPromptTemplate.from_messages([
    ('system', 'You are a helpful customer support agent.'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', '{query}')
])

# load chat history
chat_history = []
with open('chat_history.txt') as f:
    chat_history.extend(f.readlines())

# create prompt
prompt = chat_template.invoke({'chat_history': chat_history, 'query': 'Where is my refund?'})

# get response
response = model.invoke(prompt)
print(response)