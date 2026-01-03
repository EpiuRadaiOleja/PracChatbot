import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv



load_dotenv()


def document_loader(docs_path="docs"):

    if not os.path.exists(docs_path):
        raise FileNotFoundError(print(f"The directory {docs_path} not found"))
    
    loader= DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls= TextLoader,
        loader_kwargs ={"encoding":"utf-8"}
        )
    documents=loader.load()

    print(f"Loaded {len(documents)} documents.")

    if len(documents) == 0:
        raise ValueError("No documents were loaded. Please check the directory and file formats.")

    return documents

def Semantictext_splitter(documents):
    print("Splitting documents")

    embedding_model=GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

    semantic_splitter = SemanticChunker(
        embeddings=embedding_model,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount = 70
    )
    semantic_chunks = semantic_splitter.split_documents(documents)
    print("SEMANTIC CHUNKING RESULTS:")
    print("=" * 50)
    for i, chunk in enumerate(semantic_chunks, 1):
        print(f"Chunk {i}: ({len(chunk.page_content)} chars)")
        print(f'"{chunk.page_content}"')
        print()
    return semantic_chunks



def main():

    documents= document_loader(docs_path="docs")

    myChunks = Semantictext_splitter(documents)





if __name__== "__main__":
    main()