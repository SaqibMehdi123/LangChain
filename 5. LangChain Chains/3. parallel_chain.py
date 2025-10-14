from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel
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

model_gc = init_chat_model(
    "gemini-1.5-flash", 
    model_provider="google_genai", 
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

prompt1 = PromptTemplate(
    template='Generate short and simple notes from the text: {text}',
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template='Generate a 5 short question answer quiz from the text: {text}',
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template='Merge the provided notes and quiz into a single document, notes -> {notes}, quiz -> {quiz}',
    input_variables=['notes', 'quiz']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes': prompt1 | model_gc | parser,
    'quiz': prompt2 | model_hf | parser
})

merge_chain = prompt3 | model_gc | parser

chain = parallel_chain | merge_chain

text = """
Support vector machines (SVMs) are a set of supervised learning methods used for classification, regression and outliers detection.

The advantages of support vector machines are:

Effective in high dimensional spaces.

Still effective in cases where number of dimensions is greater than the number of samples.

Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.

Versatile: different Kernel functions can be specified for the decision function. Common kernels are provided, but it is also possible to specify custom kernels.

The disadvantages of support vector machines include:

If the number of features is much greater than the number of samples, avoid over-fitting in choosing Kernel functions and regularization term is crucial.

SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation (see Scores and probabilities, below).

The support vector machines in scikit-learn support both dense (numpy.ndarray and convertible to that by numpy.asarray) and sparse (any scipy.sparse) sample vectors as input. However, to use an SVM to make predictions for sparse data, it must have been fit on such data. For optimal performance, use C-ordered numpy.ndarray (dense) or scipy.sparse.csr_matrix (sparse) with dtype=float64.
"""

result = chain.invoke({'text': text})
print("\n=== Final Output (Using Parallel Chain) ===")
print(f"Type: {type(result)}")
print(f"Content: {result}")

chain.get_graph().print_ascii()