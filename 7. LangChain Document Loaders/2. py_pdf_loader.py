# PyPDFLoader
# document loader in langchain used to load content from pdf files and convert each page into a document object
# Limitation: uses the pypdf library which may not handle complex pdf layouts or extract images/tables effectively

from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv(filename=".env", usecwd=True))
print(f"GOOGLE_API_KEY loaded: {'YES' if os.getenv('GOOGLE_API_KEY') else 'NO'}")

model = GoogleGenerativeAI(
    model="gemini-2.5-pro",
    temperature=0.9,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

prompt = PromptTemplate(
    template='Write a summary for the following document: \n {text}', 
    input_variables=['text']
)

parser = StrOutputParser()

loader = PyPDFLoader('dl-curriculum.pdf')

documents = loader.load()

chain = prompt | model | parser

# result = chain.invoke({'text': documents[22].page_content})
# print(f"Summary: \n{result}")

print("Document type:", type(documents))
print(f"Number of documents: {len(documents)}")
print("Document content:\n", documents[20].page_content)
print("Metadata:", documents[20].metadata)