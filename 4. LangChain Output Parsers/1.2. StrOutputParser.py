# StrOutputParser:
# simplest output parsers in LangChain
# It is used to parse the output of an LLM and return it as a plain string

from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv, find_dotenv
import os

# Load environment variables from parent directory
dotenv_path = find_dotenv(filename=".env", usecwd=True)
load_dotenv(dotenv_path)

# Check if API key is loaded
api_key = os.getenv("GOOGLE_API_KEY")
print(f"GOOGLE_API_KEY loaded: {'YES' if api_key else 'NO'}")

model = init_chat_model(
    "gemini-1.5-flash", 
    model_provider="google_genai", 
    google_api_key=api_key
)

# 1st template --> report
template1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

# 2nd template --> summary  
template2 = PromptTemplate(
    template='Write a 5 line summary of the text: {text}',
    input_variables=['text']
)

parser = StrOutputParser()

chain1 = template1 | model | parser
report = chain1.invoke({'topic': 'Black Hole'})

chain2 = template2 | model | parser
summary = chain2.invoke({'text': report})

print("\n=== Final Output (Using StrOutputParser) ===")
print(f"Type: {type(summary)}")
print(f"Content: {summary}")