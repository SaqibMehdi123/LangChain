from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough
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
    template='write me a joke about {topic}', 
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Explain the joke in 3 lines: {text}',
    input_variables=['text']
)

joke_gen_chain = RunnableSequence(prompt1, model, parser)

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'explanation': RunnableSequence(prompt2, model, parser)
})

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

result = final_chain.invoke({'topic': 'AI'})
print(result)