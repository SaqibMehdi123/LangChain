from langchain.text_splitter import CharacterTextSplitter

text = """
Space exploration has led to incredible scientific discoveries. From landing on the Moon to exploring Mars, humanity continues to push the boundaries of what's possible beyond our planet.

These missions have not only expanded our knowledge of the universe but have also contributed to advancements in technology here on Earth. Satellite communications, GPS, and even certain medical imaging techniques trace their roots back to innovations driven by space programs.
"""

splitter = CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
    separator=""
)

chunks = splitter.split_text(text)

print(f"Total chunks created: {len(chunks)}\n")
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1} (length {len(chunk)}):\n{chunk}\n")