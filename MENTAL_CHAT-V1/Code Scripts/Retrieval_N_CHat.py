import os

#Langchain Components
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage

from dotenv import load_dotenv



load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE API KEY NOT FOUND")

CHROMA_PATH= r"E:\GitHub Repoz\AI chatBot Practice\PracChatbot\MENTAL_CHAT-V1\VecDBs"

embedding_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

llm_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

def get_VectorStore(persist_directory=CHROMA_PATH):
    Database = Chroma(
        persist_directory = persist_directory,
        embedding_function = embedding_model,
        collection_metadata = {"hnsw:space":"cosine"}

    )
    return Database


def retrieve_mmr_docs(vector_db, query, k=5, fetch_k=10, lambda_mult=0.65):
    
    #  Setup the retriever with MMR logic
    retriever = vector_db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,               # Number of documents to return
            "fetch_k": fetch_k,    # Number of documents to initially fetch for diversity check
            "lambda_mult": lambda_mult # 0.5 to 0.7 is usually the 'sweet spot'
        }
    )

    print(f"\nPerforming MMR Search for: '{query}'")

    # Invoke search
    relevant_docs = retriever.invoke(query)


    return relevant_docs

def retrieve_mmr_docs(vector_db, query, k=5, fetch_k=10, lambda_mult=0.65):
    """Retrieves diverse document chunks using MMR search"""
    retriever = vector_db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "fetch_k": fetch_k,
            "lambda_mult": lambda_mult
        }
    )
    return retriever.invoke(query)

def start_chat():

    print("Mental Health Assistant Initialized (Type 'n' to exit) ---")
    
    # Initialize database once before starting the loop
    vector_db = get_VectorStore()

    while True:
        user_query = input("\nAsk Away: ").strip()
        
        if user_query.lower() == "n":
            print("See Ya Later")
            break

        # Retrieve Context from your PDFs
        relevant_docs = retrieve_mmr_docs(vector_db, user_query)
        
        # Format the retrieved chunks into a single string
        context_text = "\n\n".join([
            f"Source: {doc.metadata.get('filename', 'Unknown')}\n{doc.page_content}" 
            for doc in relevant_docs
        ])

        #  Construct the Augmented Prompt
        messages = [
            SystemMessage(content=(
                "You are a polite Psychologist and Psychiatrist with in-depth knowledge of "
                "Mental Illness, Policies, and Diagnosis Techniques. Provide detailed and direct answers. "
                "Use the following provided context from official documents to inform your response. "
                "If the answer is not in the context, state that 'I Lack Sufficient Knowledge To address your Query'."
                f"\n\nCONTEXT FROM  DOCUMENTS:\n{context_text}"
            )),
            HumanMessage(content=user_query)
        ]

        # 5. Generate and Print Response
        try:
            result = llm_model.invoke(messages)
            print(f"\nUser: {user_query}")
            print(f"\nDR. Psycho:\n{result.content}")

              # Display Results
            if not relevant_docs:
                print(" No documents found.")
            else:
                print("\n\nSOURCES USED")
                print(f"✅ Retrieved {len(relevant_docs)} \n")
                for i, doc in enumerate(relevant_docs, 1):
                    source = doc.metadata.get('filename', 'Unknown Source')
                    print(f"Result {i} | Source: {source} ")
                    print(f"{doc.page_content}\n")

        except Exception as e:
            print(f"Error generating response: {e}")
    
    
  

if __name__ == "__main__":
    start_chat()



