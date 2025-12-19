import time
import streamlit as st

# Try imports from LangChain + Ollama; if they fail, provide safe fallbacks so the app still runs.
try:
    from langchain_community.tools import DuckDuckGoSearchRun
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough
    from langchain_ollama import ChatOllama
    LANGCHAIN_AVAILABLE = True
except Exception:
    LANGCHAIN_AVAILABLE = False

    # Minimal DuckDuckGo search fallback (uses Instant Answer API + HTML fallback if needed)
    class DuckDuckGoSearchRun:
        def run(self, q):
            try:
                import requests
            except Exception:
                return [{"snippet": f"(stub) Search result for: {q}", "url": ""}]
            try:
                params = {"q": q, "format": "json", "no_html": 1, "skip_disambig": 1}
                resp = requests.get("https://api.duckduckgo.com/", params=params, timeout=10)
                data = None
                try:
                    data = resp.json()
                except Exception:
                    data = None
                results = []
                if data:
                    abstract = (data.get("AbstractText") or "").strip()
                    if abstract:
                        results.append({"snippet": abstract, "url": data.get("AbstractURL", "")})
                    for r in data.get("Results", []):
                        txt = r.get("Text") or r.get("Result") or r.get("Snippet") or ""
                        if txt:
                            results.append({"snippet": txt, "url": r.get("FirstURL", r.get("Url", ""))})
                    for rt in data.get("RelatedTopics", []):
                        if "Text" in rt:
                            results.append({"snippet": rt.get("Text", ""), "url": rt.get("FirstURL", "")})
                        elif "Topics" in rt:
                            for sub in rt.get("Topics", [])[:3]:
                                if "Text" in sub:
                                    results.append({"snippet": sub.get("Text", ""), "url": sub.get("FirstURL", "")})
                if results:
                    # dedupe
                    seen = set(); out = []
                    for r in results:
                        s = r.get("snippet", "").strip()
                        if s and s not in seen:
                            seen.add(s); out.append(r)
                            if len(out) >= 6: break
                    if out:
                        return out
                # HTML fallback
                try:
                    from bs4 import BeautifulSoup
                except Exception:
                    return [{"snippet": f"(stub) Search result for: {q}", "url": ""}]
                headers = {"User-Agent": "Mozilla/5.0"}
                resp = requests.get("https://duckduckgo.com/html/", params={"q": q}, headers=headers, timeout=10)
                soup = BeautifulSoup(resp.text, "html.parser")
                results = []
                for a in soup.select("a.result__a, a.result-link, a[href^='/l/?kh=1&uddg=']")[:6]:
                    title = a.get_text(strip=True); url = a.get('href') or ''
                    snippet = ''
                    parent = a.find_parent()
                    if parent:
                        s = parent.select_one('.result__snippet, .snippet, p')
                        if s: snippet = s.get_text(strip=True)
                    results.append({"snippet": snippet or title, "url": url})
                if results: return results[:6]
            except Exception:
                return [{"snippet": f"(stub) Search result for: {q}", "url": ""}]
            return [{"snippet": f"(stub) Search result for: {q}", "url": ""}]

    # Minimal prompt/template/parser/run stubs
    class ChatPromptTemplate:
        @staticmethod
        def from_template(tpl):
            return tpl

    class StrOutputParser:
        def parse(self, x):
            return str(x)

    class RunnablePassthrough:
        class _Assign:
            def __init__(self, **kwargs):
                self._kwargs = kwargs
            def __or__(self, other):
                # allow piping but we will not build a full pipeline; return a stub chain
                return StubChain()
        @staticmethod
        def assign(**kwargs):
            return RunnablePassthrough._Assign(**kwargs)

    # A trivial ChatOllama stub that simply echoes or returns a formatted answer
    class ChatOllama:
        def __init__(self, model='llama3:8b', temperature=0.0):
            self.model = model; self.temperature = temperature
        def __or__(self, other):
            return StubChain()
        def __call__(self, prompt_text):
            return f"[LLM not available locally] {prompt_text[:400]}"

    # A minimal chain fallback that returns a concise summary from search results
    class StubChain:
        def invoke(self, ctx):
            q = ctx.get('question') if isinstance(ctx, dict) else str(ctx)
            search = DuckDuckGoSearchRun()
            results = search.run(q)
            if not results:
                return "I could not find any information on that."
            top = results[0]
            top_text = top.get('snippet') if isinstance(top, dict) else str(top)
            # first sentence
            import re
            sents = re.split(r'(?<=[.!?])\s+', (top_text or '').strip())
            short = sents[0].strip() if sents and sents[0] else top_text
            # sources
            sources = []
            for r in results[:3]:
                if isinstance(r, dict):
                    url = r.get('url') or r.get('snippet')
                else:
                    url = str(r)
                if url and url not in sources:
                    sources.append(url)
            src = "\n\nSources:\n" + "\n".join(f"- {u}" for u in sources) if sources else ''
            return short + src

# --- UI polish ---
st.set_page_config(page_title="Real-Time RAG Assistant", page_icon="🤖", layout="wide")

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
    show_sources = st.checkbox("Show sources section", value=True)
    st.divider()
    if st.button("🧹 Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --- LangChain components ---
if LANGCHAIN_AVAILABLE:
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

    # Build a LangChain-style pipeline if the real libs are present
    chain = (
        RunnablePassthrough.assign(
            context=lambda x: search.run(x["question"]),
            sources_rule=lambda _: sources_rule(),
        )
        | prompt
        | llm
        | StrOutputParser()
    )
else:
    # Fallback chain uses the StubChain defined above
    llm = ChatOllama(model=model, temperature=temperature)
    search = DuckDuckGoSearchRun()
    chain = StubChain()

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
            # Chain.invoke for real pipeline, or StubChain.invoke
            try:
                answer = chain.invoke({"question": user_input}) if hasattr(chain, 'invoke') else chain({"question": user_input})
            except Exception as e:
                answer = f"An error occurred while running the pipeline: {e}"
            dt = time.time() - t0
            st.write(answer)
            st.markdown(f"<div class='small'>⏱️ {dt:.1f}s</div>", unsafe_allow_html=True)

    st.session_state.messages.append({"role": "assistant", "content": answer})
