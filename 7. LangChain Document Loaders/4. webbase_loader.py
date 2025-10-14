from langchain_community.document_loaders import WebBaseLoader
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
    template='Answer the following question \n {question} from the following text - \n {text}',
    input_variables=['question','text']
)

parser = StrOutputParser()

# Wikipedia is generally more accessible for web scraping
url = 'https://en.wikipedia.org/wiki/MacBook_Air'

# Configure WebBaseLoader with better headers
loader = WebBaseLoader(
    web_paths=[url],
    header_template={
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
)

print(f"Loading content from: {url}")

try:
    documents = loader.load()
    print(f"Successfully loaded {len(documents)} document(s)")
    print(f"Content length: {len(documents[0].page_content)} characters")
    
    chain = prompt | model | parser
    
    result = chain.invoke({
        'question': 'What is MacBook Air and when was it first released?',
        'text': documents[0].page_content[:3000]
    })
    print(f"\nAnswer: {result}")
    
except Exception as e:
    print(f"Error loading website: {e}")
    print("Try with a different URL or check your internet connection.")