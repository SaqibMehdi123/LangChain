# StructuredOutputParser
# It is an output parser in langchain that helps extract structured json data from LLM based on predefined field schema
# works by defining list of fields (ResponseSchema) that the model should return, ensuring the output follows a structured format

from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
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
    temperature=0.1,
    max_new_tokens=512,
    top_p=0.95,
    repetition_penalty=1.15,
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

model = ChatHuggingFace(llm=llm)

schema = [
    ResponseSchema(name='fact_1', description='first fact about the topic'),
    ResponseSchema(name='fact_2', description='second fact about the topic'),
    ResponseSchema(name='fact_3', description='third fact about the topic')
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template='Give me 3 interesting facts about {topic} \n {format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

prompt = template.invoke({'topic': 'Black Hole'})
result = model.invoke(prompt)
print("\n=== Final Output (Using StructuredOutputParser) - Simple ===")
final_result = parser.parse(result.content)
print(f"Type: {type(final_result)}")
print(f"Content: \n{final_result}")

chain = template | model | parser
result = chain.invoke({'topic': 'Black Hole'})
print("\n=== Final Output (Using StructuredOutputParser) - Chaining ===")
print(f"Type: {type(result)}")
print(f"Content: \n{result}")