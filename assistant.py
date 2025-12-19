try:
    from langchain_community.tools import DuckDuckGoSearchRun
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser
    from langchain_ollama import ChatOllama
    LANGCHAIN_AVAILABLE = True
except Exception:
    # Fallbacks when LangChain / Ollama are not installed (avoid import-time crash)
    LANGCHAIN_AVAILABLE = False

    class DuckDuckGoSearchRun:
        def run(self, q):
            try:
                import requests
            except Exception:
                return [{"snippet": f"(stub) Search result for: {q}", "url": ""}]
            try:
                resp = requests.get("https://api.duckduckgo.com/", params={"q": q, "format": "json", "no_html": 1}, timeout=10)
                data = resp.json()
                results = []
                abstract = (data.get("AbstractText") or "").strip()
                if abstract:
                    results.append({"snippet": abstract, "url": data.get("AbstractURL", "")})
                for r in data.get("Results", []):
                    txt = r.get("Text") or r.get("Result") or r.get("Snippet") or ""
                    if txt:
                        results.append({"snippet": txt, "url": r.get("FirstURL", "")})
                for rt in data.get("RelatedTopics", [])[:6]:
                    if isinstance(rt, dict) and rt.get("Text"):
                        results.append({"snippet": rt.get("Text"), "url": rt.get("FirstURL", "")})
                if results:
                    return results[:6]
            except Exception:
                pass
            return [{"snippet": f"(stub) Search result for: {q}", "url": ""}]

    class ChatPromptTemplate:
        @staticmethod
        def from_template(tpl):
            return tpl

    class StrOutputParser:
        def parse(self, x):
            return str(x)

    class RunnablePassthrough:
        @staticmethod
        def assign(**kwargs):
            class _A:
                def __or__(self, other):
                    return StubChain()
            return _A()

    class ChatOllama:
        def __init__(self, model="llama3:8b", temperature=0.0):
            self.model = model
            self.temperature = temperature
        def __or__(self, other):
            return StubChain()
        def __call__(self, prompt):
            return f"[LLM not available] {str(prompt)[:400]}"

    class StubChain:
        def invoke(self, ctx):
            q = ctx.get("question") if isinstance(ctx, dict) else str(ctx)
            search = DuckDuckGoSearchRun()
            results = search.run(q)
            if not results:
                return "I could not find any information on that."
            top = results[0]
            text = top.get("snippet") if isinstance(top, dict) else str(top)
            import re
            sents = re.split(r"(?<=[.!?])\s+", (text or "").strip())
            short = sents[0] if sents else text
            return short

# LLM via Ollama (or fallback)
llm = None
if LANGCHAIN_AVAILABLE:
    llm = ChatOllama(model="llama3:8b", temperature=0)
else:
    llm = ChatOllama(model="llama3:8b", temperature=0)

# Web search tool (no API key)
search = DuckDuckGoSearchRun()

prompt = ChatPromptTemplate.from_template(
    """Answer with a direct, specific answer first (1 line), then 2-5 bullet points of support.
Use ONLY the search results. If missing, say "I could not find any information on that."

Search Results:
{context}

Question:
{question}
"""
)

# chain: if real RunnablePassthrough & llm exist, build chain; else fallback to stub
try:
    chain = (
        RunnablePassthrough.assign(
            context=lambda x: search.run(x["question"])
        )
        | prompt
        | llm
        | StrOutputParser()
    )
except Exception:
    chain = StubChain()

print("🤖 Hello! I'm a real-time AI assistant. Type 'exit' to quit.")
while True:
    user_query = input("You: ").strip()
    if user_query.lower() in ["exit", "quit"]:
        print("🤖 Goodbye!")
        break

    print("🤖 Thinking...")
    try:
        response = chain.invoke({"question": user_query}) if hasattr(chain, 'invoke') else chain({"question": user_query})
        print("🤖:", response)
    except Exception as e:
        print("❌ Error:", e)
        print("Tip: If it's a DuckDuckGo 'Ratelimit' error, wait a bit and try again.")
