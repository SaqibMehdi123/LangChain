# RunnableLambda is a runnable primitive that allows you to apply custom python functions within an AI pipeline.
# It acts as a middleware between different components of the pipeline, enabling preprocessing, transformation, API calls, filters, and post-processing of data in a LangChain workflow.

from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda
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

def word_counter(text):
    return len(text.split())

joke_gen_chain = RunnableSequence(prompt1, model, parser)

chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'word_count': RunnableLambda(word_counter)
})

final_chain = RunnableSequence(joke_gen_chain, chain)

result = final_chain.invoke({'topic': 'AI'})

print(f"Joke: {result['joke']}")
print(f"Word Count: {result['word_count']}")