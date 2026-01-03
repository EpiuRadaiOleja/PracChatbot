import os 

# Unstructured for document parsing
from langchain_unstructured import UnstructuredLoader
from unstructured.cleaners.core import clean_extra_whitespace

# LangChain components
from langchain_community.document_loaders import DirectoryLoader
from langchain_google_genai import  GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from dotenv import load_dotenv

load_dotenv()


def load_pdf_documents(docs_path="./MENTAL_CHAT-V1/Knowledge_Base"):
    abs_path = os.path.abspath(docs_path)
    print(f"Scanning directory: {abs_path}")

    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"The directory {abs_path} does not exist!")

    # DirectoryLoader handles the folder; UnstructuredLoader handles the PDF logic
    loader = DirectoryLoader(
        path=abs_path,
        glob="**/*.pdf",
        loader_cls=UnstructuredLoader,
        loader_kwargs={
            "strategy": "fast", # High-quality parsing
            "chunking_strategy": "by_title",
            "max_characters": 2000,
            "combine_text_under_n_chars": 500,
            "language": ["eng"],
            "post_processors":[clean_extra_whitespace]
        },
        show_progress=True
    )

    print("Partitioning documents (this may take a moment for hi_res)...")
    documents = loader.load()
    
    print(f"Successfully loaded {len(documents)} document chunks.")

    if len(documents) > 0:
        # Preview the first chunk
        print(f"\nExample Metadata: {documents[0].metadata}")
        print(f"Example Content: {documents[0].page_content[:150]}...")

    return documents

#Vector store method 
# Defining the vecDB specific absolute path
CHROMA_PATH = r"E:\GitHub Repoz\AI chatBot Practice\PracChatbot\MENTAL_CHAT-V1\VecDBs"

def create_vectorStore(chunks, persist_directory=CHROMA_PATH):
    """Creates and persists ChromaDB vector store at the specified absolute path"""
    print(f"Creating embeddings and storing in ChromaDB at: {persist_directory}")

    cleaned_chunks = filter_complex_metadata(chunks)

    # Ensure the directory exists
    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory)
        print(f"Created new directory at: {persist_directory}")

    embedding_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    
    # 2. Creating ChromaDB store
    vectorStore = Chroma.from_documents(
        documents=cleaned_chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )
    
    print("--- Finished creating vector store ---")
    print(f"Vector store is now persistent at {persist_directory}")
    return vectorStore
  



def main():

    try:
        # 1. Load, Partition, and Chunk
        docs = load_pdf_documents()
        
        # 2. Check if the list is empty before proceeding
        if not docs:
            print("No documents found. Please check your file path and PDF files.")
            return

        # 3. Print a helpful summary instead of the raw list
        print(f"\n--- Ingestion Summary ---")
        print(f"Total Chunks Created: {len(docs)}")
        
        # 4. Preview just the first chunk to verify metadata
        print(f"\n--- Preview of First Chunk ---")
        print(f"Source: {docs[0].metadata.get('source')}")
        print(f"Content: {docs[0].page_content[:250]}...") # Show first 250 chars

    except Exception as e:
        print(f"An error occurred during the main execution: {e}")

    vecDBs = create_vectorStore(chunks=docs, persist_directory=CHROMA_PATH)

if __name__ == "__main__":
    main()