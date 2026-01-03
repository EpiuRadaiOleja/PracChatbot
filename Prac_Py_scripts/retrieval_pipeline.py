import os
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
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
    "score_threshold":0.3
})

relevant_docs = retriever.invoke(query)

#Display results
for i, doc in enumerate(relevant_docs,1):
    print(f"Document{i}:\n{doc.page_content}\n")

#Answer Generation 
combined_input = f"""Based on the following documents, please answer this question: {query}

Documents:
{chr(10).join([f"- {doc.page_content}" for doc in relevant_docs])}

Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
"""

# Create a ChatGoogleGenerativeAI model
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# Define the messages for the model
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content=combined_input),
]

# Invoke the model with the combined input
result = model.invoke(messages)

# Display the full result and content only
print("\n--- Generated Response ---")
# print("Full result:")
# print(result)
print("Content only:")
print(result.content)