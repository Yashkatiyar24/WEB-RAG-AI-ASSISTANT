import time
import streamlit as st

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama

st.set_page_config(page_title="Real-Time RAG Assistant", page_icon="🤖", layout="wide")

# --- UI polish ---
st.markdown("""
<style>
.block-container { padding-top: 1.5rem; }
[data-testid="stChatMessage"] { padding: 0.6rem 0.2rem; }
.small { opacity: 0.7; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

st.title("🤖 Real-Time Web AI Assistant")
st.caption("Ollama (local LLM) + DuckDuckGo (web search) + LangChain (RAG)")

# --- Sidebar controls ---
with st.sidebar:
    st.header("⚙️ Controls")
    model = st.selectbox("Model", ["llama3:8b", "mistral", "llama3.1:8b"], index=0)
    temperature = st.slider("Creativity (temperature)", 0.0, 1.0, 0.0, 0.1)
    show_sources = st.toggle("Show sources section", value=True)
    st.divider()
    if st.button("🧹 Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --- LangChain components ---
llm = ChatOllama(model=model, temperature=temperature)
search = DuckDuckGoSearchRun()

prompt = ChatPromptTemplate.from_template(
    """You are a helpful assistant.
Answer based ONLY on the Search Results.

Rules:
- First line: short direct answer.
- Then: bullet points (max 6) with key facts.
- If Search Results are empty or irrelevant: say "I could not find any information on that."
- {sources_rule}

Search Results:
{context}

Question:
{question}
"""
)

def sources_rule():
    return "Include a 'Sources:' section with URLs if present." if show_sources else "Do not include a Sources section."

chain = (
    RunnablePassthrough.assign(
        context=lambda x: search.run(x["question"]),
        sources_rule=lambda _: sources_rule(),
    )
    | prompt
    | llm
    | StrOutputParser()
)

# --- In-memory chat history (no storage) ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render messages
for m in st.session_state.messages:
    st.chat_message(m["role"]).write(m["content"])

# Quick prompts (engaging)
st.markdown("#### 🔥 Quick prompts")
cols = st.columns(4)
suggestions = [
    "What are today's trending AI news topics?",
    "Summarize the latest updates on India politics this week.",
    "What’s the newest iPhone model and what changed?",
    "Give me 5 article ideas on democracy in India based on current news."
]
for i, s in enumerate(suggestions):
    if cols[i].button(s, use_container_width=True):
        st.session_state.prefill = s

user_input = st.chat_input("Ask anything…")

# If suggestion clicked
if "prefill" in st.session_state and st.session_state.prefill:
    user_input = st.session_state.prefill
    st.session_state.prefill = ""

# Handle new message
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Searching + thinking…"):
            t0 = time.time()
            answer = chain.invoke({"question": user_input})
            dt = time.time() - t0
            st.write(answer)
            st.markdown(f"<div class='small'>⏱️ {dt:.1f}s</div>", unsafe_allow_html=True)

    st.session_state.messages.append({"role": "assistant", "content": answer})
