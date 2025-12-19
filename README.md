# Real-Time Web AI Assistant

This repository contains a small Streamlit app (`app.py`) that runs a Real-Time RAG-style assistant: it uses web search (DuckDuckGo Instant Answer API / HTML fallback) and optionally a local or cloud LLM (Ollama or OpenAI) to summarize results.

Quick status
- The app includes robust fallbacks so it runs even when LangChain / Ollama / OpenAI packages are not installed. In fallback mode the assistant returns concise summaries and sources from DuckDuckGo search results.

Requirements
- Python 3.9+ (tested in a venv)
- See `requirements.txt` for the minimal packages needed for basic operation. Optional LLM support requires additional packages and configuration (see below).

Local run (recommended for testing)
1. Create and activate a virtualenv:

```bash
python -m venv .venv
source .venv/bin/activate  # macOS / Linux
.venv\Scripts\activate     # Windows (PowerShell)
```
2. Install dependencies:

```bash
pip install -r requirements.txt
```
3. Run Streamlit:

```bash
streamlit run app.py
```

Streamlit deploy (Cloud)
1. Push this repo to GitHub (already done).
2. Go to https://streamlit.io/cloud and connect your GitHub account.
3. Create a new app and select the repository `Yashkatiyar24/WEB-RAG-AI-ASSISTANT`, branch `main`, and file `app.py`.
4. Set environment variables if you plan to use OpenAI:
   - `OPENAI_API_KEY` - your OpenAI API key (leave unset to run in fallback/search-only mode).
5. Click Deploy. The app will run using `requirements.txt`.

Enabling LLMs (optional)
- OpenAI: Uncomment `openai` in `requirements.txt` and set `OPENAI_API_KEY` in Streamlit secrets or environment variables. The app can be extended to use OpenAI for summarization.
- Ollama (local): Run a local Ollama daemon and install the Python client packages; Streamlit Cloud cannot run Ollama locally.

Repository notes & housekeeping
- `.idea/` was previously committed; a `.gitignore` has been added and `.idea` has been removed from tracking.
- The app is intentionally resilient so the UI still works when LLM packages are missing.

If you want, I can:
- Wire OpenAI integration (requires your API key), or
- Create a `Dockerfile` for containerized deploy,
- Configure GitHub Actions to run basic lint/tests.


