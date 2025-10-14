from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
import os
import streamlit as st
from langchain.prompts import PromptTemplate, load_prompt
import time

load_dotenv()

st.header('Researcher AI Assistant')

paper_input = st.selectbox(
    "Select Research Paper Name", 
    ["Attention Is All You Need", 
     "BERT: Pre-training of Deep Bidirectional Transformers", 
     "GPT-3: Language Models are Few-Shot Learners", 
     "Diffusion Models Beat GANs on Image Synthesis"]
)

style_input = st.selectbox(
    "Select Explanation Style", 
    ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"]
)

length_input = st.selectbox(
    "Select Explanation Length", 
    ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"]
)

model = GoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

template = load_prompt('template.json')

if st.button('Summarize'):
    chain = template | model
    result = chain.invoke({
        "paper_input": paper_input,
        "style_input": style_input,
        "length_input": length_input
    })

    placeholder = st.empty()
    typed_text = ""

    for char in result:
        typed_text += char
        placeholder.write(typed_text)
        time.sleep(0.01)