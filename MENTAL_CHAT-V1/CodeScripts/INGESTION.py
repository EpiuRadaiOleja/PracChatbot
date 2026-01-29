import os
import shutil
from tqdm import tqdm
from chromadb.utils.batch_utils import create_batches
from sentence_transformers import SentenceTransformer

from langchain_community.document_loaders import DirectoryLoader, PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb



#Document loader method
def load_documents(docs_path ="source_pdfs"):
    print(f"Loading documents from {docs_path}...")

    
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} does not exist.")
    
    
    loader = DirectoryLoader(
        path = docs_path,
          glob="*.pdf", 
          loader_cls = PDFPlumberLoader,
          show_progress = True
    )

    documents = loader.load() 
    
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
        separators=["\n\n","\n",".",","," "],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    chunks = Recursivetext_splitter.split_documents(documents)

    if chunks:
        for i, chunk in enumerate(tqdm(chunks[:2])):
            print(f"\n---chunk{i+1}---")
            print(f"Source: {chunk.metadata['source']}")
            print(f"Length: {len(chunk.page_content)} characters")
            print("-"*50)

        if len(chunk.page_content)>2:
            print(f"\n.....and {len(chunks)-5} more chunks")

    return chunks

#Vector store method 
 

def create_vectorStore(chunks, persist_directory="chroma_vecStore"):

    print("Creating embeddings....")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-V2')
    client = chromadb.PersistentClient(path=persist_directory)
    collection = client.get_or_create_collection(name="pdf_collection")

    
    ids = [f"id_{i}" for i in range(len(chunks))]
    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]

    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    batches = create_batches(
        api=client, 
        ids=ids, 
        embeddings=embeddings, 
        documents=texts, 
        metadatas=metadatas
    )

    for batch in tqdm(batches):
        collection.add(
            ids=batch[0],
            embeddings=batch[1],
            metadatas=batch[2],
            documents=batch[3]
        )

    print(f"Vector store created successfully!")
    return collection

def main():
    db_path ="chroma_vecStore"

    if os.path.exists(db_path):
        print("Cleaning old DB")
        shutil.rmtree(db_path)

    print("Loading documents...")
    
    documents = load_documents(docs_path ="source_pdfs")


    chunks = split_documents(documents)

    vectorStoring = create_vectorStore(chunks)


if __name__ == "__main__":
    main()