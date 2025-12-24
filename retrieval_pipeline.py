import os
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv


load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("Error: GOOGLE_API_KEY not found! Check your .env file.")

persist_directory="chroma_db"

embedding_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

DataBase = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space":"cosine"}
)

query= input("Ask Away:")

#retriever = DataBase.as_retriever(search_kwargs={"k":3})

retriever = DataBase.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k":3,
    "score_threshold":0.5
})

relevant_docs = retriever.invoke(query)

#Display results
for i, doc in enumerate(relevant_docs,1):
    print(f"Document{i}:\n{doc.page_content}\n")
