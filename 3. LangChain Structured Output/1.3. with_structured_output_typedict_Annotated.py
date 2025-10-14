import os
from typing import TypedDict, Annotated, Optional
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

# Initialize the chat model
llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")

# Schema for the structured output
class Review(TypedDict):
    key_themes: Annotated[list[str], "A list of key themes mentioned in the review"]
    summary: Annotated[str, "A brief summary of the review"]
    sentiment: Annotated[str, "The sentiment of the review (e.g., positive, negative, neutral)"]
    pros: Annotated[Optional[list[str]], "A list of pros mentioned in the review"]
    cons: Annotated[Optional[list[str]], "A list of cons mentioned in the review"]

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
"""
)

# Try structured output, fallback to plain output if not supported
try:
    structured_model = llm.with_structured_output(Review)
    result = structured_model.invoke(text)
    print(result)
    print('Type of result:', type(result))
    print('Key Themes:', result['key_themes'])
    print('Summary:', result['summary'])
    print('Sentiment:', result['sentiment'])
    print('Pros:', result['pros'])
    print('Cons:', result['cons'])
except NotImplementedError:
    print("Structured output is not supported for this model. Showing plain output:")
    result = llm.invoke(text)
    print(result)