import os
from pathlib import Path

# Unstructured for document parsing
from langchain_unstructured import UnstructuredLoader
from unstructured.cleaners.core import clean_extra_whitespace
from langchain_community.document_loaders import CSVLoader

# LangChain components
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from dotenv import load_dotenv

load_dotenv()

DOCUMENT_PATH = r"E:\GitHub Repoz\AI chatBot Practice\PracChatbot\MENTAL_CHAT-V1\Knowledge_Base"
CHROMA_PATH = r"E:\GitHub Repoz\AI chatBot Practice\PracChatbot\MENTAL_CHAT-V1\VecDBs"

def load_pdf_documents(docs_path=DOCUMENT_PATH):
    """Load PDF documents using UnstructuredLoader with chunking by title."""
    abs_path = os.path.abspath(docs_path)
    print(f"Scanning directory: {abs_path}")

    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"The directory {abs_path} does not exist!")

    # Get all PDF files in the directory and subdirectories
    pdf_files = list(Path(abs_path).rglob("*.pdf")) + list(Path(abs_path).rglob("*.PDF"))
    all_documents = []
    
    print(f"Found {len(pdf_files)} PDF files to process...")
    
    for pdf_file in pdf_files:
        print(f"Processing: {pdf_file.name}")
        
        try:
            # Load each PDF file individually with UnstructuredLoader
            loader = UnstructuredLoader(
                file_path=str(pdf_file),
                strategy="hi_res",  # High-quality parsing
                chunking_strategy="by_title",
                max_characters=2000,
                combine_text_under_n_chars=500,
                languages=["eng"],
                post_processors=[clean_extra_whitespace]
            )
            
            documents = loader.load()
            all_documents.extend(documents)
            
        except Exception as e:
            print(f"Error processing {pdf_file}: {e}")
            continue

    print(f"Successfully loaded {len(all_documents)} document chunks from {len(pdf_files)} PDF files.")
    return all_documents

def load_csv_documents(docs_path=DOCUMENT_PATH):
    """Load CSV documents using CSVLoader."""
    abs_path = os.path.abspath(docs_path)
    
    # Get all CSV files in the directory and subdirectories
    csv_files = list(Path(abs_path).rglob("*.csv")) + list(Path(abs_path).rglob("*.CSV"))
    all_documents = []
    
    print(f"Found {len(csv_files)} CSV files to process...")
    
    for csv_file in csv_files:
        print(f"Processing: {csv_file.name}")
        
        try:
            # Load CSV file
            loader = CSVLoader(file_path=str(csv_file))
            documents = loader.load()
            all_documents.extend(documents)
            
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
            continue

    print(f"Successfully loaded {len(all_documents)} document chunks from {len(csv_files)} CSV files.")
    return all_documents

def load_all_documents(docs_path=DOCUMENT_PATH):
    """Load both PDF and CSV documents."""
    print("Loading PDF documents...")
    pdf_docs = load_pdf_documents(docs_path)
    
    print("Loading CSV documents...")
    csv_docs = load_csv_documents(docs_path)
    
    # Combine all documents
    all_documents = pdf_docs + csv_docs
    
    print(f"Total documents loaded: {len(all_documents)}")
    
    if len(all_documents) > 0:
        # Preview the first chunk
        print(f"\nExample Metadata: {all_documents[0].metadata}")
        print(f"Example Content: {all_documents[0].page_content[:150]}...")

    return all_documents

def create_vectorStore(chunks, persist_directory=CHROMA_PATH):
    """Creates and persists ChromaDB vector store at the specified absolute path"""
    print(f"Creating embeddings and storing in ChromaDB at: {persist_directory}")

    cleaned_chunks = filter_complex_metadata(chunks)

    # Ensure the directory exists
    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory)
        print(f"Created new directory at: {persist_directory}")

    embedding_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    
    # Creating ChromaDB store
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
        # 1. Load all documents (PDFs and CSVs)
        docs = load_all_documents()
        
        # 2. Check if the list is empty before proceeding
        if not docs:
            print("No documents found. Please check your file path and files.")
            return

        # 3. Print a helpful summary
        print(f"\n--- Ingestion Summary ---")
        print(f"Total Chunks Created: {len(docs)}")
        
        # 4. Preview just the first chunk to verify metadata
        if docs:
            print(f"\n--- Preview of First Chunk ---")
            # Handle different metadata keys for PDF vs CSV
            first_doc = docs[0]
            source = (first_doc.metadata.get('filename') or 
                     first_doc.metadata.get('source') or 
                     first_doc.metadata.get('path', 'Unknown'))
            print(f"Source: {source}")
            print(f"Content: {first_doc.page_content[:250]}...")  # Show first 250 chars

        vecDBs = create_vectorStore(chunks=docs, persist_directory=CHROMA_PATH)
        print("Vector store creation completed successfully!")

    except Exception as e:
        print(f"An error occurred during the main execution: {e}")
        import traceback
        traceback.print_exc()  # Print full traceback for debugging

if __name__ == "__main__":
    main()