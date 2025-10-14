from langchain_community.document_loaders import TextLoader
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
    template='Write a summary for the following poem: \n {text}', 
    input_variables=['text']
)

parser = StrOutputParser()

loader = TextLoader('ai_poem.txt', encoding='utf-8')

documents = loader.load()

# print("Document type:", type(documents))
# print(f"Number of documents: {len(documents)}")
# print("Document content:", documents[0].page_content)
# print("Metadata:", documents[0].metadata)

chain = prompt | model | parser

result = chain.invoke({'text': documents[0].page_content})
print(f"Summary: \n{result}")