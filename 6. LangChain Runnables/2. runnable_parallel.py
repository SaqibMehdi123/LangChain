from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableParallel
from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv(filename=".env", usecwd=True))
print(f"GOOGLE_API_KEY loaded: {'YES' if os.getenv('GOOGLE_API_KEY') else 'NO'}")

model = GoogleGenerativeAI(
    model="gemini-2.5-pro",
    temperature=0.9,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template='Generate a tweet about {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a linkedin post about {topic}',
    input_variables=['topic']
)

parallel_chain = RunnableParallel({
    'tweet': RunnableSequence(prompt1, model, parser),
    'linkedin_post': RunnableSequence(prompt2, model, parser)
})

result = parallel_chain.invoke({'topic': 'AI'})
print(result)