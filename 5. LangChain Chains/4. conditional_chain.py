from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain.schema.runnable import RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv(filename=".env", usecwd=True))
api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
print(f"HUGGINGFACEHUB_API_TOKEN loaded: {'YES' if api_key else 'NO'}")
print(f"GOOGLE_API_KEY loaded: {'YES' if os.getenv('GOOGLE_API_KEY') else 'NO'}")

llm_hf = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    temperature=0.1,
    max_new_tokens=512,
    top_p=0.95,
    repetition_penalty=1.15,
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

model_hf = ChatHuggingFace(llm=llm_hf)

class Feedback(BaseModel):
    sentiment: Literal['Positive', 'Negative'] = Field(..., description="The sentiment of the feedback")

parser_pydantic = PydanticOutputParser(pydantic_object=Feedback)
parser = StrOutputParser()

prompt1 = PromptTemplate(
    template='Classify the sentiment of the following text as Positive or Negative: {feedback}. {format_instructions}',
    input_variables=['feedback'],
    partial_variables={'format_instructions': parser_pydantic.get_format_instructions()}
)

prompt2 = PromptTemplate(
    template='Write an appropriate response to this positive feedback: {feedback}',
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template='Write an appropriate response to this negative feedback: {feedback}',
    input_variables=['feedback']
)

classifier_chain = prompt1 | model_hf | parser_pydantic

branch_chain = RunnableBranch(
    (lambda x: x.sentiment == 'Positive', prompt2 | model_hf | parser),
    (lambda x: x.sentiment == 'Negative', prompt3 | model_hf | parser),
    RunnableLambda(lambda x: 'Could not find Sentiment')
)

chain = classifier_chain | branch_chain

result = chain.invoke({'feedback': 'This is a beautiful smartphone.'})

print('=== Classification Output (Using PydanticOutputParser) ===')
print('Type:', type(result))
print(result)

chain.get_graph().print_ascii()