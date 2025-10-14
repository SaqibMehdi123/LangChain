# PydanticOutputParser:
# PydanticOutputParser is a structured output parser in langchain that uses Pydantic models to enforce schema validation when processing LLM responses.
# strict schema enforcement, type safety, easy validation, seamless integration

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from dotenv import load_dotenv, find_dotenv
import os

# Load environment variables from parent directory
dotenv_path = find_dotenv(filename=".env", usecwd=True)
load_dotenv(dotenv_path)

# Check if API key is loaded
api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
print(f"HUGGINGFACEHUB_API_TOKEN loaded: {'YES' if api_key else 'NO'}")

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    temperature=0.9,
    max_new_tokens=512,
    top_p=0.95,
    repetition_penalty=1.15,
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

model = ChatHuggingFace(llm=llm)

class Person(BaseModel):
    name: str = Field(description='Name of the person')
    age: int = Field(gt=18, description='Age of the person')
    city: str = Field(description='Name of the city the person lives in')

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template='Generate the name, age, and city of a fictional {place} person \n {format_instruction}',
    input_variables=['place'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

prompt = template.invoke({'place': 'Pakistani'})
result = model.invoke(prompt)
final_result = parser.parse(result.content)
print("\n=== Final Output (Using PydanticOutputParser) - Simple ===")
print(f"Type: {type(final_result)}")
print(f"Content: \n{final_result}")

chain = template | model | parser
result = chain.invoke({'place': 'Pakistani'})
print("\n=== Final Output (Using PydanticOutputParser) - Chaining ===")
print(f"Type: {type(result)}")
print(f"Content: \n{result}")