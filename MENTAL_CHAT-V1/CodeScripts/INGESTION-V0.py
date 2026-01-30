
import os
import shutil
import pandas as pd
from tqdm import tqdm
from chromadb.utils.batch_utils import create_batches
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb




# Document loader method with metadata
def load_documents_with_metadata(docs_path="source_pdfs", meta_csv="source_pdfs/src_pdfs.csv"):
    print(f"Loading documents from {docs_path} with metadata from {meta_csv}...")

    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} does not exist.")
    if not os.path.exists(meta_csv):
        raise FileNotFoundError(f"The metadata CSV {meta_csv} does not exist.")

    meta = pd.read_csv(meta_csv)
    if 'PDF NAME' not in meta.columns:
        raise ValueError("The metadata CSV must have a 'PDF NAME' column.")

    pdf_files = [f for f in os.listdir(docs_path) if f.endswith('.pdf')]
    meta_pdfs = meta['PDF NAME'].tolist()
    matching_pdfs = [pdf for pdf in meta_pdfs if pdf in pdf_files]

    if not matching_pdfs:
        raise ValueError("No matching PDFs found between directory and metadata CSV.")

    documents = []
    for pdf_file in tqdm(matching_pdfs, desc="Loading PDFs"):
        pdf_path = os.path.join(docs_path, pdf_file)
        try:
            with PDFPlumberLoader(pdf_path) as loader:
                loaded_docs = loader.load()
                # Attach metadata from CSV to each document
                file_meta = meta[meta['PDF NAME'] == pdf_file].iloc[0].to_dict()
                for doc in loaded_docs:
                    doc.metadata.update(file_meta)
                documents.extend(loaded_docs)
        except Exception as e:
            print(f"Failed to load {pdf_file}: {e}")

    print(f"Loaded {len(documents)} documents.")
    if len(documents) == 0:
        raise ValueError("No documents were loaded. Please check the directory and file formats.")

    for i, doc in enumerate(tqdm(documents[:2])):  # show preview of first 2 documents
        print(f"\nDocument {i+1}:")
        print(f" Source: {doc.metadata.get('source', doc.metadata.get('PDF NAME', 'N/A'))}")
        print(f" Length: {len(doc.page_content)} characters")
        print(f" Content Preview: {doc.page_content[:100]}...")
        print(f" metadata: {doc.metadata}")

    return documents
 

# Chunking method (metadata is already attached)
def split_documents(documents, chunk_size=1000, chunk_overlap=10):
    print("Splitting Documents into chunks....")
    Recursivetext_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", ",", " "],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = Recursivetext_splitter.split_documents(documents)
    if chunks:
        for i, chunk in enumerate(tqdm(chunks[:2])):
            print(f"\n---chunk{i+1}---")
            print(f"Source: {chunk.metadata.get('source', chunk.metadata.get('PDF NAME', 'N/A'))}")
            print(f"Length: {len(chunk.page_content)} characters")
            print("-"*50)
        if len(chunks) > 5:
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
    db_path = "chroma_vecStore"
    meta_csv = "source_pdfs/src_pdfs.csv"
    if os.path.exists(db_path):
        print("Cleaning old DB")
        shutil.rmtree(db_path)
    print("Loading documents with metadata...")
    documents = load_documents_with_metadata(docs_path="source_pdfs", meta_csv=meta_csv)
    chunks = split_documents(documents)
    vectorStoring = create_vectorStore(chunks)

if __name__ == "__main__":
    main()