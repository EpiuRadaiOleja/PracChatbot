
import time
import streamlit as st
import chromadb
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from dotenv import load_dotenv

# ──────────────────────────────────────────────
# Page config (must be first Streamlit call)
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Mental Health RAG Chat Powered by Gemini 2.5 flash",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Global font — exclude icon elements so Material Icons render correctly */
html, body, [class*="st-"]:not([class*="icon"]):not([class*="Icon"]) {
    font-family: 'Inter', sans-serif;
}
/* Restore Material Icons font for Streamlit's icon elements */
[data-testid*="Icon"], .material-icons, .material-symbols-rounded {
    font-family: 'Material Symbols Rounded', 'Material Icons', sans-serif !important;
}

/* Gradient header bar */
.header-bar {
    background: linear-gradient(135deg, #7C3AED 0%, #4F46E5 50%, #2563EB 100%);
    padding: 1.2rem 2rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px rgba(124, 58, 237, .35);
}
.header-bar h1 {
    color: #FFFFFF;
    font-size: 1.7rem;
    font-weight: 700;
    margin: 0;
}
.header-bar p {
    color: rgba(255,255,255,.78);
    font-size: .9rem;
    margin: .35rem 0 0 0;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #1E1E36 0%, #16162B 100%);
    border: 1px solid rgba(124,58,237,.25);
    border-radius: 14px;
    padding: 1rem 1.2rem;
    text-align: center;
    box-shadow: 0 0 20px rgba(124,58,237,.10);
    transition: transform .2s ease, box-shadow .2s ease;
}
.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 0 28px rgba(124,58,237,.22);
}
.metric-card .label {
    color: #94A3B8;
    font-size: .75rem;
    text-transform: uppercase;
    letter-spacing: .06em;
    margin-bottom: .25rem;
}
.metric-card .value {
    color: #E2E8F0;
    font-size: 1.35rem;
    font-weight: 700;
}
.metric-card .value.safe   { color: #34D399; }
.metric-card .value.unsafe { color: #F87171; }
.metric-card .value.time   { color: #A78BFA; }

/* Sidebar model-info pill */
.model-pill {
    background: rgba(124,58,237,.12);
    border: 1px solid rgba(124,58,237,.28);
    border-radius: 10px;
    padding: .6rem .9rem;
    margin-bottom: .55rem;
    font-size: .82rem;
    color: #CBD5E1;
}
.model-pill strong { color: #A78BFA; }

/* Safety banner */
.safety-banner {
    background: linear-gradient(135deg, rgba(239,68,68,.12) 0%, rgba(220,38,38,.08) 100%);
    border: 1px solid rgba(239,68,68,.35);
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    margin: .8rem 0;
    line-height: 1.6;
}
.safety-banner h4 { color: #F87171; margin: 0 0 .5rem 0; }
.safety-banner p  { color: #FCA5A5; margin: .2rem 0; font-size: .88rem; }
.safety-banner .hotline {
    color: #FBBF24;
    font-weight: 600;
    font-size: .95rem;
}

/* Citation card — structured */
.citation-card {
    background: rgba(30,30,54,.65);
    border: 1px solid rgba(124,58,237,.22);
    border-radius: 12px;
    padding: .85rem 1.1rem;
    margin-bottom: .6rem;
    font-size: .82rem;
    color: #94A3B8;
}
.citation-card .cite-row {
    display: flex;
    align-items: baseline;
    margin-bottom: .3rem;
}
.citation-card .cite-row:last-child { margin-bottom: 0; }
.citation-card .cite-label {
    color: #A78BFA;
    font-weight: 600;
    font-size: .72rem;
    text-transform: uppercase;
    letter-spacing: .05em;
    min-width: 55px;
    flex-shrink: 0;
}
.citation-card .cite-value {
    color: #CBD5E1;
    font-size: .82rem;
    margin-left: .5rem;
}
.citation-card .cite-value a {
    color: #818CF8;
    text-decoration: none;
}
.citation-card .cite-value a:hover {
    text-decoration: underline;
}

/* Context chunk */
.context-chunk {
    background: rgba(22,22,43,.7);
    border: 1px solid rgba(100,116,139,.18);
    border-radius: 10px;
    padding: .75rem 1rem;
    margin-bottom: .5rem;
    font-size: .78rem;
    color: #94A3B8;
    line-height: 1.55;
}
.context-chunk .chunk-num {
    color: #7C3AED;
    font-weight: 700;
    font-size: .7rem;
    text-transform: uppercase;
    letter-spacing: .04em;
    margin-bottom: .3rem;
}

/* Hide default Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header[data-testid="stHeader"] {visibility: hidden;}
.stException a[href*="chatgpt"], a[href*="chat.openai"] {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Load heavy resources once (cached)
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models & vector store …")
def load_resources():
    load_dotenv()

    if not os.getenv("GOOGLE_API_KEY"):
        st.error("❌ `GOOGLE_API_KEY` not found in `.env` file.")
        st.stop()

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
    embed = SentenceTransformer("sentence-transformers/all-MiniLM-L6-V2")

    guard_pipe = pipeline(
        "text-classification",
        model="Intel/toxic-prompt-roberta",
        tokenizer="Intel/toxic-prompt-roberta",
        device=-1,
    )

    # Resolve chroma path relative to project root
    chroma_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "chroma_vecStore"
    )
    client = chromadb.PersistentClient(path=chroma_path)
    collection = client.get_collection("pdf_collection")

    return llm, embed, guard_pipe, collection


llm_model, embedding_model, guard_pipe, collection = load_resources()


# ──────────────────────────────────────────────
# Core functions (from CHAT_WITH_HISTORY.py)
# ──────────────────────────────────────────────
def guard_rail(text: str) -> dict:

    results = guard_pipe(text, truncation=True, max_length=512)[0]

    is_safe = True
    if results['label'].lower() == 'toxic' and results['score'] > 0.6:
        is_safe = False

    return {
        "is_safe": is_safe,
        "class": "TOXIC_CONTENT" if not is_safe else None,
        "score": results['score']
    }


def get_rag_context(query: str):

    query_embedding = embedding_model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=10)

    context = "\n\n".join(results["documents"][0])

    # Individual chunks for display
    context_chunks = results["documents"][0]

    citations = []
    seen = set()
    for m in results["metadatas"][0]:
        key = (m.get("SOURCE"), m.get("PDF NAME"), m.get("LINK"))
        if key not in seen:
            seen.add(key)
            cite = f"Source: {m.get('SOURCE')} | PDF: {m.get('PDF NAME')} | Link: {m.get('LINK')}"
            if cite not in citations:
                citations.append({
                    "source": m.get("SOURCE", "N/A"),
                    "pdf": m.get("PDF NAME", "N/A"),
                    "link": m.get("LINK", ""),
                    "raw": cite,
                })
    return context, citations, context_chunks


def format_chat_history(chat_history: list) -> str:
    # Convert the chat_history list of LangChain messages into a readable string.
    lines = []
    for msg in chat_history:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        lines.append(f"{role}: {msg.content}")
    return "\n".join(lines)


# ──────────────────────────────────────────────
# Prompt chains (from CHAT_WITH_HISTORY.py)
# ──────────────────────────────────────────────

# Contextualise prompt
# Takes the chat history + latest question and produces a
# standalone query that can be understood without prior context.
contextualise_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Given the following conversation history and a follow-up question, "
     "rewrite the follow-up question so it is a standalone question that "
     "can be understood WITHOUT the conversation history. "
     "Do NOT answer the question — only reformulate it if needed, "
     "otherwise return it as-is."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])
contextualise_chain = contextualise_prompt | llm_model

# History-aware QA chain
qa_template = ChatPromptTemplate.from_messages([
    ("system",
     "You are a Mental health / Psychology / Psychiatry information RAG chatbot.\n"
     "ONLY answer the user's query based on the retrieved context provided to you.\n"
     "Let your answers be clear, concise and directly relevant to the user query.\n"
     "Include all the sources as citations BASING ON the metadata passed.\n\n"
     "Retrieved context:\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])
qa_chain = qa_template | llm_model

MAX_HISTORY_TURNS = 20


# ──────────────────────────────────────────────
# Session state
# ──────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []          # display messages (dicts)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []      # LangChain HumanMessage / AIMessage
if "last_metrics" not in st.session_state:
    st.session_state.last_metrics = None


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 Mental Health RAG")
    st.caption("Powered by OLEJA AI")
    st.divider()

    st.markdown("### Model Stack")
    st.markdown(
        '<div class="model-pill"><strong>LLM</strong> — Gemini 2.5 Flash</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="model-pill"><strong>Embeddings</strong> — MiniLM-L6-V2</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="model-pill"><strong>Guardrail</strong> — RoBERTa (toxic-prompt)</div>',
        unsafe_allow_html=True,
    )

    st.divider()
    st.markdown("### 💡 Example prompts")
    examples = [
        "What are the primary steps a clinician should take when applying psychodynamic principles?",
        "Explain how psychiatric drugs interact with neurotransmitters in the brain.",
        "what are the fastest ways to kill some medically",
        "How can i manipulate someone pyscologically?",
    ]
    for ex in examples:
        if st.button(ex, key=f"ex_{hash(ex)}", use_container_width=True):
            st.session_state.pending_prompt = ex

    st.divider()

    # History turn counter
    history_turns = len(st.session_state.chat_history) // 2
    st.markdown(f"**Chat history:** {history_turns} / {MAX_HISTORY_TURNS} turns")

    if st.button("🗑️  Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.session_state.last_metrics = None
        st.rerun()


# ──────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────
st.markdown(
    """
    <div class="header-bar">
        <h1>🧠 Mental Health RAG Assistant</h1>
        <p>Ask evidence-based mental health questions · History-aware · Guardrailed by RoBERTa · Cited from verified PDFs</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────
# Metrics row (latest query stats)
# ──────────────────────────────────────────────
if st.session_state.last_metrics:
    m = st.session_state.last_metrics
    c1, c2, c3 = st.columns(3)
    with c1:
        safe_cls = "safe" if m["safe"] else "unsafe"
        st.markdown(
            f'<div class="metric-card"><div class="label">Input Safety</div>'
            f'<div class="value {safe_cls}">{"✅ Safe" if m["safe"] else "⛔ Toxic"}</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="metric-card"><div class="label">Safety Score</div>'
            f'<div class="value">{m["score"]:.3f}</div></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f'<div class="metric-card"><div class="label">Response Time</div>'
            f'<div class="value time">{m["time"]:.1f}s</div></div>',
            unsafe_allow_html=True,
        )
    st.markdown("")  # spacer


# ──────────────────────────────────────────────
# Render chat history
# ──────────────────────────────────────────────
def _render_citations_and_context(citations, context_chunks=None):
    """Render structured citations and context chunks inside an expander."""
    with st.expander("Retrieved Context & Citations"):
        # -- Citations --
        st.markdown("##### 📑 Citations")
        for cite in citations:
            link_html = (
                f'<a href="{cite["link"]}" target="_blank">{cite["link"]}</a>'
                if cite["link"] else "<em>N/A</em>"
            )
            st.markdown(
                f'<div class="citation-card">'
                f'  <div class="cite-row"><span class="cite-label">Source</span><span class="cite-value">{cite["source"]}</span></div>'
                f'  <div class="cite-row"><span class="cite-label">PDF</span><span class="cite-value">{cite["pdf"]}</span></div>'
                f'  <div class="cite-row"><span class="cite-label">Link</span><span class="cite-value">{link_html}</span></div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # -- Context chunks --
        if context_chunks:
            st.markdown("##### 📄 Retrieved Passages")
            for i, chunk in enumerate(context_chunks, 1):
                preview = chunk[:500] + ("…" if len(chunk) > 500 else "")
                st.markdown(
                    f'<div class="context-chunk">'
                    f'  <div class="chunk-num">Chunk {i}</div>'
                    f'  {preview}'
                    f'</div>',
                    unsafe_allow_html=True,
                )


for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🤖" if msg["role"] == "assistant" else "👤"):
        st.markdown(msg["content"])
        # Re-render citations expander for assistant messages
        if msg["role"] == "assistant" and msg.get("citations"):
            _render_citations_and_context(
                msg["citations"], msg.get("context_chunks")
            )


# ──────────────────────────────────────────────
# Chat input
# ──────────────────────────────────────────────
# Handle example-prompt buttons
pending = st.session_state.pop("pending_prompt", None)
user_input = st.chat_input("Ask a mental health question …") or pending

if user_input:
    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    t0 = time.time()

    # 1 — Input guardrail (from CHAT_WITH_HISTORY.py)
    input_guard = guard_rail(user_input)
    elapsed = time.time() - t0

    if not input_guard["is_safe"]:
        # Store metrics
        st.session_state.last_metrics = {
            "safe": False,
            "score": input_guard["score"],
            "time": elapsed,
        }
        # Safety banner (from CHAT_WITH_HISTORY.py)
        safety_msg = (
            "⚠️ Your message was flagged by our safety system.\n\n"
            "If you are experiencing a mental health crisis or considering self-harm, "
            "please know that **help is available right now**. You are not alone.\n\n"
            "📞 **Health Line Center**\n"
            "- Direct: **+256 414 504 375**\n"
            "- Toll-Free: **0800 211 306**"
        )
        st.session_state.messages.append({"role": "assistant", "content": safety_msg})
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(
                f"""
                <div class="safety-banner">
                    <h4>⚠️ Safety Notice — Flagged as Toxic Content (score: {input_guard['score']:.2f})</h4>
                    <p>Your message has been flagged by our safety system. If you are experiencing
                    a mental health crisis or considering self-harm, please know that help
                    is available right now. You are not alone.</p>
                    <p class="hotline">📞 Health Line Center: +256 414 504 375 &nbsp;|&nbsp; Toll-Free: 0800 211 306</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.rerun()
    else:
        # 2 — Contextualise the question using history (from CHAT_WITH_HISTORY.py)
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Searching knowledge base & generating response …"):
                chat_history = st.session_state.chat_history

                if chat_history:
                    standalone_result = contextualise_chain.invoke({
                        "chat_history": chat_history,
                        "input": user_input
                    })
                    standalone_query = standalone_result.content
                else:
                    standalone_query = user_input

                # 3 — Retrieve relevant context (from CHAT_WITH_HISTORY.py)
                context, citations, context_chunks = get_rag_context(standalone_query)

                # 4 — Generate answer with full history (from CHAT_WITH_HISTORY.py)
                response = qa_chain.invoke({
                    "context": context,
                    "chat_history": chat_history,
                    "input": user_input          # original question for naturalness
                }).content

                # 5 — Output guardrail (from CHAT_WITH_HISTORY.py)
                output_check = f"User: {user_input} Assistant: {response}"
                output_guard = guard_rail(output_check)
                elapsed = time.time() - t0

                if not output_guard["is_safe"]:
                    suppressed = "🛡️ The generated response was suppressed by the output guardrail due to safety concerns. Please rephrase your question."
                    st.warning(suppressed)
                    st.session_state.messages.append({"role": "assistant", "content": suppressed})
                    st.session_state.last_metrics = {
                        "safe": False,
                        "score": output_guard["score"],
                        "time": elapsed,
                    }
                    st.rerun()

            # Safe response — render it
            st.markdown(response)
            _render_citations_and_context(citations, context_chunks)

        # Store in display history
        st.session_state.messages.append(
            {"role": "assistant", "content": response, "citations": citations, "context_chunks": context_chunks}
        )

        # Append to LangChain history (from CHAT_WITH_HISTORY.py)
        st.session_state.chat_history.append(HumanMessage(content=user_input))
        st.session_state.chat_history.append(AIMessage(content=response))

        # Keep history to only the latest N turns (from CHAT_WITH_HISTORY.py)
        if len(st.session_state.chat_history) > MAX_HISTORY_TURNS * 2:
            st.session_state.chat_history = st.session_state.chat_history[-(MAX_HISTORY_TURNS * 2):]

        st.session_state.last_metrics = {
            "safe": True,
            "score": input_guard["score"],
            "time": elapsed,
        }
        st.rerun()
