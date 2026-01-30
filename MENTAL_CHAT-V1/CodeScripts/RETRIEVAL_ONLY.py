import os
import chromadb
from sentence_transformers import SentenceTransformer



embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-V2')

persist_directory = "chroma_vecStore"
client = chromadb.PersistentClient(path=persist_directory)
collection = client.get_collection("pdf_collection")
print(f"Total embeddings in collection: {collection.count()}\n\n")


def retriever(query, top_k=10):
    query_embedding = embedding_model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=top_k)
    retrieved_ids = results["ids"][0]
    retrieved_texts = results["documents"][0]

    return retrieved_ids, retrieved_texts

def retrieval_basedon_query(query, n_results=5):
    
    retrieved_ids, retrieved_texts = retriever(query, top_k=10)
    response = "\n\n".join(retrieved_texts[:n_results])

    return response

def retrieved_content():

    while True:
        user_query = input("User: ")
        if user_query == "exit":
            print("Byeeee.......")
            break
        
        response = retrieval_basedon_query(user_query)
        print(response)

if __name__ == "__main__":
    retrieved_content()