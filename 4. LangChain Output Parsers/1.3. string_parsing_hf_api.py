# StrOutputParser:
# simplest output parsers in LangChain
# It is used to parse the output of an LLM and return it as a plain string

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv, find_dotenv
import os

# Load environment variables from parent directory
dotenv_path = find_dotenv(filename=".env", usecwd=True)
load_dotenv(dotenv_path)

# Check if API key is loaded
api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
print(f"HUGGINGFACEHUB_API_TOKEN loaded: {'YES' if api_key else 'NO'}")

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    temperature=0.1,
    max_new_tokens=512,
    top_p=0.95,
    repetition_penalty=1.15,
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

model = ChatHuggingFace(llm=llm)

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