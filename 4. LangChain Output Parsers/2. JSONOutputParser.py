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

parser = JsonOutputParser()

template = PromptTemplate(
    template='Give me the name, age and city of a fictional person \n {format_instruction}',
    input_variables=[],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

prompt = template.format()
result = model.invoke(prompt)
print("\n=== Final Output (Using JsonOutputParser) - Simple ===")
final_result = parser.parse(result.content)
print(final_result)
print(f"Type: {type(final_result)}")

 # --------------------------------------------------

chain = template | model | parser
result = chain.invoke({})
print("\n=== Final Output (Using JsonOutputParser) - Chaining ===")
print(f"Type: {type(result)}")
print(f"Content: {result}")