import os

#Langchain Components
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()


CHROMA_PATH = r"E:\GitHub Repoz\AI chatBot Practice\PracChatbot\MENTAL_CHAT-V1\VecDBs"


#          INITIALIZING EMBEDDING MODEL
embedding_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )


#        INITIALIZING API KEY CHECK AND LLM MODEL
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE API KEY NOT FOUND")

llm_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")



#                  VECTORE STORE GETTER FUNC
def get_VectorStore(persist_directory=CHROMA_PATH):
    Database = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"}
    )
    return Database

#             MMR LOGIC RETRIEVER FUNC
def retrieve_mmr_docs(vector_db, query, k=20, fetch_k=50, lambda_mult=0.65):
    retriever = vector_db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "fetch_k": fetch_k,
            "lambda_mult": lambda_mult
        }
    )
    return retriever.invoke(query)


#             MAIN FUNC
def start_chat():
    print("Mental Health Assistant Ready (Type 'n' to exit) ---")
    
    # Initialize database once before starting the loop
    try:
        vector_db = get_VectorStore()
        print("Database loaded successfully.")
    except Exception as e:
        print(f"Error loading database: {e}")
        return

    while True:
        user_query = input("\nAsk Away: ").strip()
        
        if user_query.lower() == "n":
            print("See Ya Later")
            break
        
        if not user_query:  # Handle empty input
            print("Please enter a query.")
            continue

        try:
            # Retrieve Context from your PDFs
            relevant_docs = retrieve_mmr_docs(vector_db, user_query)
            
            # Format the retrieved chunks into a single string
            context_text = "\n\n".join([
                f"Source: {doc.metadata.get('filename', 'Unknown')}\n{doc.page_content}" 
                for doc in relevant_docs
            ])

            # Constructing thee necessay combined prompt, with a Guardrail in the system Message
            messages = [
                SystemMessage(content=(
                    f"""You are an empathetic, supportive, and non-judgmental AI Mental Health Assistant IN UGANDA. Your goal is to provide information and emotional support based strictly on the provided context.

                You are NOT a doctor, therapist, or licensed mental health professional. You are an AI information retrieval system.
                
                ### PHASE 1: SAFETY ASSESSMENT (CRITICAL)
                Before processing the context or generating a response, you must analyze the user's input for "Red Flag" indicators.

                **Red Flags include:**
                1.  **Self-Harm/Suicide:** Explicit or implicit mentions of wanting to die, hurt oneself, or ending it all.
                2.  **Harm to Others:** Threats of violence or abuse toward others.
                3.  **Abuse:** Mentions of domestic violence, child abuse, or sexual assault.
                4.  **Medical Emergency:** Symptoms requiring immediate ER attention (e.g., overdose, heart attack).

                **IF A RED FLAG IS DETECTED:**
                1.  Ignore the RAG Context.
                2.  Do NOT attempt to counsel the user.
                3.  Immediately output a Crisis Response using this format:
                    *   A brief, empathetic validation statement (e.g., "I am hearing that you are in a lot of pain right now, and I am concerned for your safety.")
                    *   A directive to seek immediate help.
                    *   Your help should be tailor for a Ugandan client
                    

                ### PHASE 2: RAG OPERATIONAL RULES
                If no Red Flags are detected, proceed to answer the user query using the provided {context_text}.

                1.  **Strict Grounding:** You must answer the user's question using *only* the information provided in the `[CONTEXT]`.
                2.  **No External Knowledge:** If the answer is not in the `[CONTEXT]`, state: "I don't have information on that specific topic in my current database, but I recommend discussing this with a professional." Do not hallucinate or make up medical advice.
                3.  **Citations:** When providing coping strategies or definitions, implicitly reference the source material provided in the context (e.g., "According to the provided resources...").

                ### PHASE 3: CLINICAL GUARDRAILS
                1.  **No Diagnosis:** Never tell a user they "have" a condition. Use bridging language: "Symptoms like these are often associated with..." or "The text suggests that..."
                2.  **No Prescription:** Do not recommend specific medications or dosages, even if mentioned in the context. You may list medications *as examples of common treatments* mentioned in the text, but always add: "Please consult a psychiatrist for medication management."
                3.  **Disclaimer:** If the user asks for a professional opinion, explicitly state: "I am an AI, not a healthcare professional."

                ### PHASE 4: TONE AND STYLE
                *   **Empathetic but Objective:** Be warm and validating, but maintain professional boundaries. Avoid overly flowery or dramatic language.
                *   **Concise:** Flash models are fast; keep your answers structured (use bullet points) and readable.
                *   **Validation:** Start responses by validating the user's intent (e.g., "It is understandable to feel that way," or "Thank you for asking about this.")"""
                    
                )),
                HumanMessage(content=user_query)
            ]

            # Generate and Print Response
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
                    print(f"{doc.page_content[:500]}...")  # Limit content display for readability

        except Exception as e:
            print(f"Error generating response: {e}")

if __name__ == "__main__":
    start_chat()