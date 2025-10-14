import os
from typing import Optional, Literal
from pydantic import BaseModel, Field
from dotenv import load_dotenv, find_dotenv
from langchain.chat_models import init_chat_model

dotenv_path = find_dotenv(filename=".env", usecwd=True) 
load_dotenv(dotenv_path, override=True)

key = os.getenv("GOOGLE_API_KEY") 
print("GOOGLE_API_KEY loaded:", "YES," + key[:6] + "..." if key else "NO")

# Initialize the chat model
llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai", google_api_key=key)

# Schema for the structured output
json_schema = {
  "title": "Review",
  "type": "object",
  "properties": {
    "key_themes": {
      "type": "array",
      "items": {
        "type": "string"
      },
      "description": "Write down all the key themes discussed in the review in a list"
    },
    "summary": {
      "type": "string",
      "description": "A brief summary of the review"
    },
    "sentiment": {
      "type": "string",
      "enum": ["positive", "negative", "neutral"],
      "description": "Return sentiment of the review either positive, negative or neutral"
    },
    "pros": {
      "type": ["array", "null"],
      "items": {
        "type": "string"
      },
      "description": "Write down all the pros inside a list. Each pro should be a separate string in the array."
    },
    "cons": {
      "type": ["array", "null"],
      "items": {
        "type": "string"
      },
      "description": "Write down all the cons inside a list. Each con should be a separate string in the array."
    },
    "name": {
      "type": ["string", "null"],
      "description": "Write the name of the reviewer"
    }
  },
  "required": ["key_themes", "summary", "sentiment"]
}

text = (
    """I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.
The S-Pen integration is a great touch for note-taking and quick sketches, though I don’t use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.
However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung’s One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.

Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful

Cons:
Bulky and heavy—not great for one-handed use
Bloatware still exists in One UI
Expensive compared to competitors

Review by Saqib Mehdi
"""
)

# Try structured output, fallback to plain output if not supported
try:
    structured_model = llm.with_structured_output(json_schema)
    result = structured_model.invoke(text)
    print(result)
    print('Type of result:', type(result))
    print('Key Themes:', result['key_themes'])
    print('Summary:', result['summary'])
    print('Sentiment:', result['sentiment'])
    print('Pros:', result['pros'])
    print('Cons:', result['cons'])
    print('Name:', result['name'])
except NotImplementedError:
    print("Structured output is not supported for this model. Showing plain output:")
    result = llm.invoke(text)
    print(result)