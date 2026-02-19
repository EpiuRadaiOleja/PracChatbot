import os
import torch
import chromadb
import time
import psutil
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from sentence_transformers import SentenceTransformer
from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv()

# hf_token= os.getenv("HF_TOKEN")

# if hf_token:
#     login(token=hf_token)
#     print("Successful HuggingFace login")
# else:
#     print("Failed Login")



llm_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-V2')

# *******************************************************
def get_ram_usage():
    return psutil.virtual_memory().percent

print(f"Initial RAM Usage: {get_ram_usage()}%")
start_load=time.perf_counter()
# ********************************************************

#Guardrail: Llama-Guard-3-1B 
guard_model_id = "meta-llama/Llama-Guard-3-1B"
guard_tokenizer = AutoTokenizer.from_pretrained(guard_model_id)
guard_model = AutoModelForCausalLM.from_pretrained(
    guard_model_id, 
    torch_dtype=torch.bfloat16, 
    low_cpu_mem_usage=True, 
    device_map="auto"
)

# ***********************************************
end_load = time.perf_counter()
print(f"End RAM Usage: {get_ram_usage()}%")
# ***********************************************

print(f"Model Load Time: {end_load-start_load:.2f} seconds ")
client = chromadb.PersistentClient(path="chroma_vecStore")
collection = client.get_collection("pdf_collection")

def moderate(text):
    """Checks if text is safe. Returns True if safe, False if malicious."""
    inputs = guard_tokenizer([text], return_tensors="pt").to("cpu")
    output = guard_model.generate(**inputs, max_new_tokens=10)
    response = guard_tokenizer.decode(output[0], skip_special_tokens=True)
    return "unsafe" not in response.lower()

def get_rag_context(query):
    query_embedding = embedding_model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=5)
    
    context = "\n\n".join(results["documents"][0])
    
    # Deduplicate citations
    citations = []
    for m in results["metadatas"][0]:
        cite = f"Source: {m.get('SOURCE')} | PDF: {m.get('PDF NAME')} | Link: {m.get('LINK')}"
        if cite not in citations:
            citations.append(cite)
    return context, citations

def secure_chat():
    template = """
    You are a Mental Health RAG bot. Answer the query using ONLY the context provided.
    Context: {context}
    Query: {user_query}
    Answer:
    """
    chain = ChatPromptTemplate.from_template(template) | llm_model

    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ["exit", "quit"]: break

        # GUARDRAIL 1: Input Check
        if not moderate(user_input):
            print("Guardrail Warning: Malicious or unsafe prompt detected.")
            continue

        # RETRIEVAL
        context, citations = get_rag_context(user_input)

        # GENERATION
        response = chain.invoke({"context": context, "user_query": user_input}).content

        # GUARDRAIL 2: Output Check
        if not moderate(response):
            print("Guardrail Warning: The response was flagged as unsafe and hidden.")
        else:
            print(f"\nAssistant: {response}")
            print("\n--- REFERENCES ---")
            for c in citations:
                print(c)

if __name__ == "__main__":
    secure_chat()