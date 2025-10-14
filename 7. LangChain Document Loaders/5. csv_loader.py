from langchain_community.document_loaders import CSVLoader
import time

loader = CSVLoader(file_path='Social_Network_Ads.csv', encoding='utf-8')
documents = loader.load()

length = len(documents)
print('Document count:', length)

start = time.time()
for i in range(length):
    print(f"\nDocument {i+1} content:\n", documents[i].page_content)
    print("Metadata:", documents[i].metadata)

end = time.time()
print(f"\nTime taken to load and print documents: {(end - start)*1000} ms")