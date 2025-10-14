from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
import time

loader = DirectoryLoader(
    path='Books',
    glob='*.pdf',
    loader_cls=PyPDFLoader
)

lazy_documents = loader.lazy_load()
# documents = list(lazy_documents)

start = time.time()
for doc in lazy_documents:
    if doc.metadata['page_label'] == '345':
        print("Document content of page 345:\n", doc.page_content)
        print("Metadata:", doc.metadata)
        break
    # print(type(doc.metadata['page']))

end = time.time()
print(f"Time taken to load documents lazily: {end - start} seconds")
