from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Newer Ollama integration package
from langchain_ollama import ChatOllama

# LLM via Ollama
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


chain = (
    RunnablePassthrough.assign(
        context=lambda x: search.run(x["question"])
    )
    | prompt
    | llm
    | StrOutputParser()
)

print("🤖 Hello! I'm a real-time AI assistant. Type 'exit' to quit.")
while True:
    user_query = input("You: ").strip()
    if user_query.lower() in ["exit", "quit"]:
        print("🤖 Goodbye!")
        break

    print("🤖 Thinking...")
    try:
        response = chain.invoke({"question": user_query})
        print("🤖:", response)
    except Exception as e:
        print("❌ Error:", e)
        print("Tip: If it's a DuckDuckGo 'Ratelimit' error, wait a bit and try again.")
