from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents
    
def chunking(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(documents)

    text_chunks = [chunk.page_content for chunk in chunks]

    with open("chunks.txt", "w", encoding="utf-8") as f:
        for i, chunk in enumerate(text_chunks):
            f.write(f"Chunk {i+1}:\n{chunk}\n\n{'='*50}\n\n")

    return text_chunks


