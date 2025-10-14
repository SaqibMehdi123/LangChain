from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('dl-curriculum.pdf')
documents = loader.load()

splitter = CharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=0,
    separator=''
)

chunks = splitter.split_documents(documents)

print(f"Total chunks created: {len(chunks)}\n")

# for i, chunk in enumerate(chunks):
#     print(f"Chunk {i+1} (length {len(chunk.page_content)}):\n{chunk.page_content}\n")

print("Sample chunk content:\n", chunks[10].page_content)