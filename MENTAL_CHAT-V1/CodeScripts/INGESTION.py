import os
import chromadb
import shutil
from tqdm import tqdm

from sentence_transformers import SentenceTransformer

from langchain_community.document_loaders import DirectoryLoader, PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

#Document loader method
def load_documents(docs_path ="source_pdfs"):
    print(f"Loading documents from {docs_path}...")

    #Check if the directory exists
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} does not exist.")
    
    # Load all text files from the specified directory
    loader = DirectoryLoader(
        path = docs_path,
          glob="*.pdf", 
          loader_cls = PDFPlumberLoader,
          show_progress = True
    )

    documents = loader.load() # Load documents  
    
    print(f"Loaded {len(documents)} documents.")

    if len(documents) == 0:
        raise ValueError("No documents were loaded. Please check the directory and file formats.")

    for i, doc in enumerate(tqdm(documents[:2])): #show preview of first 2 documents
        print(f"\nDocument {i+1}:")  
        print(f" Source: {doc.metadata['source']}")
        print(f" Length: {len(doc.page_content)} characters")
        print(f" Content Preview: {doc.page_content[:100]}...")  # Print first 200 characters
        print(f" metadata: {doc.metadata}")

    return documents
 
#Chunking method

def split_documents(documents, chunk_size= 500, chunk_overlap=0):

    print("Splitting Documents into chunks....")

    Recursivetext_splitter = RecursiveCharacterTextSplitter(
        separator=["\n\n","\n",".",","," "],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    # Split the documents
    chunks = Recursivetext_splitter.split_documents(documents)

    if chunks:# If chunks were created, print some info
        for i, chunk in enumerate(tqdm(chunks[:2])):
            print(f"\n---chunk{i+1}---")
            print(f"Source: {chunk.metadata['source']}")
            print(f"Length: {len(chunk.page_content)} characters")
            print("-"*50)

        if len(chunk.page_content)>2:
            print(f"\n.....and {len(chunks)-5} more chunks")

    return chunks

#Vector store method 
def create_vectorStore(chunks,persist_directory="chroma_vecStore"):
    
    print("Creating embeddings and storing in ChromaDB")

    embedding_model = SentenceTransformer('sentence-transfomers/all-MiniLM-L6-V2')

    #Creating CgromaDB store
    with tqdm(total=1, desc="Processing Vector Store") as pbar:
        vectorStore = chromadb.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory=persist_directory,
            collection_metadata={"hnsw:space":"cosine"}
        )
    
    pbar.update(1)

    print(f"Vector store created and stored to {persist_directory}")
    return vectorStore

def main():
    db_path ="chroma_vecStore"

    if os.path.exists(db_path):
        print("Cleaning old DB")
        shutil.rmtree(db_path)
        
    print("Loading documents...")
    
    documents = load_documents(docs_path ="source_pdfs")


    #Chunking section
    chunks = split_documents(documents)

    #creating and storing in chromaDB
    vectorStoring = create_vectorStore(chunks)


if __name__ == "__main__":
    main()