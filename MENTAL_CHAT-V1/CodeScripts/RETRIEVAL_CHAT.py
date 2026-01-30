import os
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv

load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("API KEY NOT FOUND")

llm_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
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
    context = "\n\n".join(retrieved_texts[:n_results])

    return context

template="""

You are a Mental health/Psycology/Psychiatry information RAG chat bot.
ONLY answer the user's query based on the retrieved context provided to you.

Query: {user_query}

Here is the retrieved context: {context}

Answer:
Let your answers be clear, concise and directly relevant to the user query


"""

prompt = ChatPromptTemplate.from_template(template)

chain = prompt | llm_model



def chat():


    while True:
        user_query = input("User: ")
        if user_query.lower() == "exit":
            print("Byeeee.......")
            break
        
        context = retrieval_basedon_query(user_query)
        
       
        result = chain.invoke({"user_query":user_query, "context": context})
        print(f"Psycho: {result.content}")

if __name__ == "__main__":
    chat()