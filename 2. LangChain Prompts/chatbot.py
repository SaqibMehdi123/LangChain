from langchain_google_genai import GoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
import os

load_dotenv()

model = GoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

chat_history = [
    SystemMessage(content="You are a helpful assistant that answers questions in an interactive way.")
]

while True:
    user_input = input("User: ")
    if user_input.strip().lower() in ["exit", "quit"]:
        print("Exiting the chatbot. Goodbye!")
        break

    chat_history.append(HumanMessage(content=user_input))
    ai_message = model.invoke(chat_history)
    chat_history.append(AIMessage(content=ai_message))
    print("AI:", ai_message)

print(chat_history)