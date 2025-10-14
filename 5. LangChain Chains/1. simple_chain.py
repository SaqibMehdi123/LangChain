from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv(filename=".env", usecwd=True))
api_key = os.getenv("GOOGLE_API_KEY")
print(f"GOOGLE_API_KEY loaded: {'YES' if api_key else 'NO'}")

model = init_chat_model(
    "gemini-1.5-flash", 
    model_provider="google_genai", 
    google_api_key=api_key
)

prompt = PromptTemplate(
    template='Generate 5 interesting facts about {topic}',
    input_variables=['topic']
)

parser = StrOutputParser()

chain = prompt | model | parser
result = chain.invoke({'topic': 'Cricket'})
print("\n=== Final Output (Using StrOutputParser) ===")
print(f"Type: {type(result)}")
print(f"Content: {result}")

chain.get_graph().print_ascii()