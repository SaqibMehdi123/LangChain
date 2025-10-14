# RunnableBranch is a control flow component in LangChain that allows for conditional execution of different Runnables based on a specified condition.
# It functions like an if/elif/else block for chains -- where you define a set of conditon functions, each associated with a runnable (e.g. LLM call, prompt chain, or tool). The first matching condition is executed. If no condition matches, a default runnable is used (if provided).

# RunnableLambda is a runnable primitive that allows you to apply custom python functions within an AI pipeline.
# It acts as a middleware between different components of the pipeline, enabling preprocessing, transformation, API calls, filters, and post-processing of data in a LangChain workflow.

from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda, RunnableBranch
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
    template='write a detailed report about {topic}', 
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Summarize the following text \n {text}',
    input_variables=['text']
)

report_gen_chain = RunnableSequence(prompt1, model, parser)

branch_chain = RunnableBranch(
    (lambda x: len(x.split())>300, RunnableSequence(prompt2, model, parser)),
    RunnablePassthrough()
)

final_chain = RunnableSequence(report_gen_chain, branch_chain)
result = final_chain.invoke({'topic': 'Pakistan vs. India cricket match'})
print(f"Report: \n{result}")