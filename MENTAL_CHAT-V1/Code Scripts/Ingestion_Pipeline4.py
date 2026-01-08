import os
import hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Disable Chroma telemetry
os.environ["CHROMA_TELEMETRY"] = "false"
os.environ['HF_HUB_OFFLINE'] = '1'

from langchain_unstructured import UnstructuredLoader
from unstructured.cleaners.core import clean_extra_whitespace
from langchain_community.document_loaders import CSVLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.documents import Document

from sentence_transformers import SentenceTransformer


#                       CONFIG 
DOCUMENT_PATH = r"E:\GitHub Repoz\AI chatBot Practice\PracChatbot\MENTAL_CHAT-V1\Knowledge_Base"
CHROMA_PATH = r"E:\GitHub Repoz\AI chatBot Practice\PracChatbot\MENTAL_CHAT-V1\VecDBs"
EMBEDDING_MODEL_PATH= r"C:\Users\hp\.cache\huggingface\hub\models--sentence-transformers--all-MiniLM-L6-v2\snapshots\c9745ed1d9f207416be6d2e6f8de32d1f16199bf"

MAX_WORKERS = 4



#                  EMBEDDINGS INITIALIZATION
EMBEDDINGS = HuggingFaceEmbeddings(
    model_name= EMBEDDING_MODEL_PATH,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)



#                       FOR DOC HASHING
def calculate_document_hash(doc: Document) -> str:
    source = doc.metadata.get("source") or doc.metadata.get("filename", "")
    return hashlib.md5((doc.page_content + source).encode()).hexdigest()



#                         HANDLING EXISTING DBs
def get_existing_document_hashes():
    if not os.path.exists(CHROMA_PATH):
        return set()

    try:
        db = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=EMBEDDINGS,
        )

        results = db._collection.get(include=["documents", "metadatas"])

        hashes = set()
        for content, meta in zip(results["documents"], results["metadatas"]):
            source = meta.get("source") or meta.get("filename", "")
            hashes.add(hashlib.md5((content + source).encode()).hexdigest())

        return hashes

    except Exception:
        return set()



#                         LOADERS 
def process_pdf(pdf_path: Path):
    try:
        loader = UnstructuredLoader(
            file_path=str(pdf_path),
            strategy="fast",  # ✅ FIX: no OCR / no gibberish
            chunking_strategy="by_title",
            max_characters=2000,
            combine_text_under_n_chars=500,
            languages=["eng"],
            post_processors=[clean_extra_whitespace],
        )
        return loader.load()
    except Exception:
        return []


def process_csv(csv_path: Path):
    # ✅ FIX: encoding fallback
    for encoding in ("utf-8", "cp1252", "latin-1"):
        try:
            loader = CSVLoader(
                file_path=str(csv_path),
                encoding=encoding,
            )
            return loader.load()
        except Exception:
            continue
    return []



#                       DOCUMENT INGESTION 
def load_documents():
    base = Path(DOCUMENT_PATH).resolve()

    pdfs = list(base.rglob("*.pdf")) + list(base.rglob("*.PDF"))
    csvs = list(base.rglob("*.csv")) + list(base.rglob("*.CSV"))

    documents = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []

        for pdf in pdfs:
            futures.append(executor.submit(process_pdf, pdf))
        for csv in csvs:
            futures.append(executor.submit(process_csv, csv))

        for future in as_completed(futures):
            documents.extend(future.result())

    return documents



#                       REMOVING DUPLICATE DBs 
def deduplicate_documents(documents):
    existing_hashes = get_existing_document_hashes()
    seen_hashes = set()

    unique_docs = []

    for doc in documents:
        h = calculate_document_hash(doc)
        if h in existing_hashes or h in seen_hashes:
            continue
        seen_hashes.add(h)
        unique_docs.append(doc)

    return unique_docs
# 


#                          CREATING VECTOR STORE 
def create_vector_store(documents):
    if not documents:
        print("No new documents to add.")
        return

    os.makedirs(CHROMA_PATH, exist_ok=True)

    cleaned_docs = filter_complex_metadata(documents)

    Chroma.from_documents(
        documents=cleaned_docs,
        embedding=EMBEDDINGS,
        persist_directory=CHROMA_PATH,
        collection_metadata={"hnsw:space": "cosine"},
    )

    print(f"Vector store updated with {len(cleaned_docs)} documents.")



#                          MAIN FUNC
def main():
    docs = load_documents()

    if not docs:
        print("No documents found.")
        return

    unique_docs = deduplicate_documents(docs)

    if not unique_docs:
        print("All documents already exist in the vector store.")
        return

    create_vector_store(unique_docs)


if __name__ == "__main__":
    main()
