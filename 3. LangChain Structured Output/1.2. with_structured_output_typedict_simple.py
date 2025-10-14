import os
from typing import TypedDict
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

# Initialize the chat model
llm = init_chat_model(
    "gemini-2.5-flash", 
    model_provider="google_genai",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# Schema for the structured output
class Review(TypedDict):
    summary: str
    sentiment: str

text = (
    "The hardware is great, but the software feels bloated. "
    "There are too many pre-installed apps that I can’t remove. "
    "Also, the UI looks outdated compared to other brands. "
    "Hoping for a software update to fix this."
)

# Try structured output, fallback to plain output if not supported
try:
    structured_model = llm.with_structured_output(Review)
    result = structured_model.invoke(text)
    print(result)
    print('Type of result:', type(result))
    print('Summary:', result['summary'])
    print('Sentiment:', result['sentiment'])
except NotImplementedError:
    print("Structured output is not supported for this model. Showing plain output:")
    result = llm.invoke(text)
    print(result)