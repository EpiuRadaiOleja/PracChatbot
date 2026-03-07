import torch
import chromadb

from transformers import pipeline
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from sentence_transformers import SentenceTransformer

from dotenv import load_dotenv





load_dotenv()

llm_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-V2')

# --- Guardrail Initialization ---
# Using a common RoBERTa-based toxicity model
guard_model_id = "Intel/toxic-prompt-roberta" 

guard_pipe = pipeline(
        "text-classification", 
        model=guard_model_id, 
        tokenizer=guard_model_id,
        device=-1 # Returns scores for all labels
    )

client = chromadb.PersistentClient(path="chroma_vecStore")
collection = client.get_collection("pdf_collection")

def guard_rail(text: str) -> dict:
    """
    Evaluates the toxicity of the input text using RoBERTa.
    Note: Roberta models usually take strings, not chat lists.
    """
    results = guard_pipe(text)[0]
    
    # Toxicity threshold (adjust as needed, typically 0.5 to 0.7)
    THRESHOLD = 0.6 
    
    # Identify the highest scoring 'toxic' label
    is_safe = True
    flagged_label = None
    
    for entry in results:
        # Most toxicity models use labels like: toxic, severe_toxic, threat, insult, etc.
        if entry['label'].lower() in ['toxic', 'severe_toxic', 'threat', 'insult', 'identity_hate']:
            if entry['score'] > THRESHOLD:
                is_safe = False
                flagged_label = entry['label']
                break

    return {
        "is_safe": is_safe,
        "class": flagged_label,
        "scores": results
    }

def get_rag_context(query):
    query_embedding = embedding_model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=5)
    
    context = "\n\n".join(results["documents"][0])
    
    citations = []
    for m in results["metadatas"][0]:
        cite = f"Source: {m.get('SOURCE')} | PDF: {m.get('PDF NAME')} | Link: {m.get('LINK')}"
        if cite not in citations:
            citations.append(cite)
    return context, citations

def secure_chat():
    # Updated mapping for RoBERTa labels
    LABEL_MAP = {
        "toxic": "Toxic Content",
        "severe_toxic": "Severely Toxic Content",
        "threat": "Physical Threat",
        "insult": "Insult/Harassment",
        "identity_hate": "Hate Speech"
    }

    template = """
        You are a Mental health/Psychology/Psychiatry information RAG chat bot.
        ONLY answer the user's query based on the retrieved context provided to you.

        Here is the retrieved context: {context}

        Query: {user_query}
        Answer:
        Let your answers be clear, concise and directly relevant to the user query.
        Include all the sources as references/citation.
    """
    chain = ChatPromptTemplate.from_template(template) | llm_model

    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ["exit", "quit"]: break

        # Input Check - Roberta takes the string directly
        guarded_result = guard_rail(user_input)

        if not guarded_result["is_safe"]:
            reason_text = LABEL_MAP.get(guarded_result["class"], "Safety Policy Violation")
            
            print(f"\n\n### IMPORTANT SAFETY NOTICE (Flagged: {reason_text}) ###")
            print("""
            Your message has been flagged by our safety system. If you are experiencing 
            a mental health crisis or considering self-harm, please know that help 
            is available right now. You are not alone.
                  
            For immediate assistance, please reach out to the Health Line Center:
            Direct Line: +256 414 504 375
            Toll-Free:   0800 211 306

            If you are in immediate physical danger, please contact your local 
            emergency services or go to the nearest hospital.
            """)
            continue

        # Proceed with RAG
        context, citations = get_rag_context(user_input)
        response = chain.invoke({"context": context, "user_query": user_input}).content

        # Output Check (Concatenate user + assistant for context)
        output_to_check = f"User: {user_input}\nAssistant: {response}"
        output_guard_result = guard_rail(output_to_check)

        if not output_guard_result["is_safe"]:
            print(f"Guardrail Warning (Unclean output): The response was flagged as {output_guard_result['class']}.")
            continue

        print(f"\nAssistant: {response}")
        print("\n--- REFERENCES ---")
        for c in citations:
            print(c)

if __name__ == "__main__":
    secure_chat()