from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence
from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv(filename=".env", usecwd=True))
print(f"GOOGLE_API_KEY loaded: {'YES' if os.getenv('GOOGLE_API_KEY') else 'NO'}")


model = GoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.9,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

prompt1 = PromptTemplate(
    template='write me a joke about {topic}', 
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Explain the joke in 3 lines: {text}',
    input_variables=['text']
)

parser = StrOutputParser()

chain = RunnableSequence(prompt1, model, parser, prompt2, model, parser)

result = chain.invoke({'topic': 'AI'})
print('Content:', result)