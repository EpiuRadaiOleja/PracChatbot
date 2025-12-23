import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

def load_documents(docs_path ="docs"):
    print(f"Loading documents from {docs_path}...")

    #Check if the directory exists
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} does not exist.")
    
    # Load all text files from the specified directory
    loader = DirectoryLoader(
        path = docs_path,
          glob="*.txt", 
          loader_cls=TextLoader
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
 
def main():
    print("Loading documents...")
    documents = load_documents(docs_path ="docs")

# Now this is OUTSIDE the function and will run automatically
if __name__ == "__main__":
    main()