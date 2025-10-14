# Simple Output Parsing (without StrOutputParser):
# Directly access .content attribute from model response
# This is the simplest way to get string output from LangChain models

from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate
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

# Step 1: Generating detailed report
prompt1 = template1.invoke({'topic': 'Black Hole'})

result1 = model.invoke(prompt1)

# Step 2: Generating summary
prompt2 = template2.invoke({'text': result1.content})

result2 = model.invoke(prompt2)

print("\n=== Final Output (Simple String Parsing) ===")

final_output = result2.content
print(f"Type: {type(final_output)}")
print(f"Content: {final_output}")