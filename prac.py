import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

#Document loader method
def load_documents(docs_path ="docs"):
    print(f"Loading documents from {docs_path}...")

    #Check if the directory exists
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} does not exist.")
    
    # Load all text files from the specified directory
    loader = DirectoryLoader(
        path = docs_path,
          glob="*.txt", 
          loader_cls=TextLoader,
          loader_kwargs={"encoding": "utf-8"}
    )

    documents = loader.load() # Load documents  
    
    print(f"Loaded {len(documents)} documents.")

    if len(documents) == 0:
        raise ValueError("No documents were loaded. Please check the directory and file formats.")

    for i, doc in enumerate(documents[:2]): #show preview of first 2 documents
        print(f"\nDocument {i+1}:")  
        print(f" Source: {doc.metadata['source']}")
        print(f" Length: {len(doc.page_content)} characters")
        print(f" Content Preview: {doc.page_content[:100]}...")  # Print first 200 characters
        print(f" metadata: {doc.metadata}")

    return documents
 
#Chunking method

def split_documents(documents, chunk_size= 500, chunk_overlap=0):
    """"Split documents into smaller chunks with overlap """

    print("Splitting Documents into chunks....")

    Recursivetext_splitter = RecursiveCharacterTextSplitter(# Set chunk size and overlap
        separator=["\n\n","\n",".",","," "],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = Recursivetext_splitter.split_documents(documents)# Split the documents

    if chunks:# If chunks were created, print some info
        for i, chunk in enumerate(chunks[:2]):
            print(f"\n---chunk{i+1}---")
            print(f"Source: {chunk.metadata['source']}")
            print(f"Length: {len(chunk.page_content)} characters")
            print("-"*50)

        if len(chunk.page_content)>2:
            print(f"\n.....and {len(chunks)-5} more chunks")

    return chunks

#Vector store method 
def create_vectorStore(chunks,persist_directory="chroma_db"):
    """Creates and persist ChromaDB vector store"""
    print("Creating embeddings and storing in ChromaDB")

    embedding_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    #Creating CgromaDB store
    vectorStore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space":"cosine"}
    )
    print("---Finished creating vector store---")
    print(f"Vector store created and stored to {persist_directory}")
    return vectorStore

def main():

    # Load documents from the 'docs' directory
    print("Loading documents...")
    documents = load_documents(docs_path ="docs")

    #Chunking section
    chunks = split_documents(documents)

    #creating and storing in chromaDB
    vectorStoring = create_vectorStore(chunks)

# Now this is OUTSIDE the function and will run automatically
if __name__ == "__main__":
    main()