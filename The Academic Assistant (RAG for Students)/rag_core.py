import os
import json
import chromadb

# --- LangChain & Google Imports ---
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader

# ---
# 1. PERSISTENT DISK PATHS (for Render)
# ---
# Path for the single, shared Chroma database
DB_PATH = "./persistent_chroma_db"
# Path for the folder holding all user-specific chat histories
CHAT_HISTORY_DIR = "./user_chats"

# ---
# 2. RAG CORE LOGIC
# ---

def load_single_document(file_path: str, original_filename: str):
    """Loads a single document and sets its 'source' metadata to the original_filename."""
    _, extension = os.path.splitext(file_path)
    if extension == ".pdf":
        loader = PyPDFLoader(file_path)
    elif extension == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")
    elif extension == ".docx":
        loader = Docx2txtLoader(file_path)
    else:
        print(f"Warning: Unsupported file type '{extension}'. Skipping.")
        return []
    
    docs = loader.load()
    
    for doc in docs:
        doc.page_content = doc.page_content.encode('utf-8', 'ignore').decode('utf-8')
        doc.metadata["source"] = original_filename
        
    return docs

def get_or_create_vector_store(api_key: str, collection_name: str) -> Chroma:
    """
    Loads/creates a vector store for a SPECIFIC user (collection).
    """
    print(f"Loading/creating persistent vector store at: {DB_PATH} for collection: {collection_name}")
    
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=api_key
    )
    
    # KEY CHANGE:
    # We use ONE persistent directory, but a DIFFERENT collection_name for each user.
    vector_store = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings,
        collection_name=collection_name  # <-- This segregates user data
    )
    
    print("Vector store loaded successfully.")
    return vector_store

def get_processed_files_from_store(vector_store: Chroma) -> list[str]:
    """
    Retrieves the list of 'source' filenames from the user's collection.
    (The vector_store object is already user-specific)
    """
    try:
        all_metadata = vector_store.get(include=["metadatas"])['metadatas']
        if not all_metadata:
            return []
        unique_sources = set(meta.get('source') for meta in all_metadata if meta.get('source'))
        return sorted(list(unique_sources))
    except Exception as e:
        print(f"Error getting processed files: {e}")
        return []

def add_documents_to_store(
    vector_store: Chroma, 
    file_paths_with_original_names: dict, 
    api_key: str
) -> Chroma:
    """
    Loads, splits, and adds new documents to the user's collection.
    (The vector_store object is already user-specific)
    """
    print(f"Adding {len(file_paths_with_original_names)} new file(s) to user's collection.")
    
    all_documents = []
    for temp_path, original_name in file_paths_with_original_names.items():
        all_documents.extend(load_single_document(temp_path, original_name))

    if not all_documents:
        raise ValueError("No new documents were successfully loaded.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(all_documents)

    if not chunks:
        raise ValueError("New document splitting resulted in no chunks.")

    vector_store.add_documents(chunks)
    print("Successfully added new documents and persisted changes.")
    return vector_store

def get_rag_response(vector_store: Chroma, question: str, api_key: str) -> dict:
    """
    Generates a RAG response using the user's specific collection.
    (The vector_store object is already user-specific)
    """
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    prompt_template = """You are an expert academic assistant. Your task is to provide a clear and concise answer to the question based strictly on the provided context from the documents.

    Follow these instructions:
    1.  Analyze the context provided below. The context is extracted from various documents (like PDFs, PowerPoints, Word documents, or text files).
    2.  Formulate a comprehensive answer to the question using only the information found in the context.
    3.  If the context does not contain the information needed to answer the question, respond with: "I'm sorry, but the provided documents do not contain enough information to answer this question."
    4.  Structure your answer in a clear, easy-to-read format. Use bullet points, numbered lists, or bold text to highlight key information where appropriate.
    5.  Do not add any information that is not present in the context.

    **Context:**
    {context}

    **Question:**
    {question}

    **Answer:**
    """
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    retrieved_docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)

    if not context.strip():
        print("Warning: No context retrieved for the question.")
        pass

    final_prompt_string = prompt.format(context=context, question=question)

    model = GoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.9,
        google_api_key=api_key
    )

    result = model.invoke(final_prompt_string) 
    
    return {
        "answer": result,
        "sources": retrieved_docs
    }

# ---
# 3. CHAT HISTORY MANAGEMENT (Multi-User)
# ---

def get_user_chat_file(username: str) -> str:
    """Returns the path to the user's specific chat history file."""
    os.makedirs(CHAT_HISTORY_DIR, exist_ok=True) 
    return f"{CHAT_HISTORY_DIR}/{username}_history.json"

def create_default_chat():
    """Returns the structure for a new, empty chat history."""
    return {
        "chat_0": {
            "title": "New Chat",
            "messages": [{"role": "assistant", "content": "Welcome! I've loaded your persistent database. Upload new documents or ask questions about existing ones."}]
        },
        "chat_id_counter": 1
    }

def load_chat_history(username: str):
    """Loads a specific user's chat history from their JSON file."""
    chat_file = get_user_chat_file(username)
    if os.path.exists(chat_file):
        try:
            with open(chat_file, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return create_default_chat()
    else:
        return create_default_chat()

def save_chat_history(history, username: str):
    """Saves a specific user's chat history to their JSON file."""
    chat_file = get_user_chat_file(username)
    with open(chat_file, "w") as f:
        json.dump(history, f, indent=4)

# ---
# 4. TITLE GENERATION
# ---

def generate_chat_title(prompt: str, api_key: str) -> str:
    """Generates a concise 2-word title for a chat."""
    try:
        model = GoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
            temperature=0.3
        )
        title_prompt = f"""
        Generate a concise chat title of 2 words or less based on the following user query. 
        Only return the title itself, with no "Title:" prefix or quotes.
        User Query: "{prompt}"
        Title:
        """
        response = model.invoke(title_prompt)
        return response.strip().replace('"', '')
    except Exception as e:
        print(f"Error in title generation: {e}")
        return "New Chat"