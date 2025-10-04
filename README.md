clinical-codes-finder

A lightweight FastAPI app that lets you search across major clinical vocabularies — ICD-10-CM, LOINC, RxTerms, HCPCS, UCUM, and HPO — returning top-ranked codes, grouped by coding system, along with a short, plain-English summary explaining why each code is relevant.

✨ Features

🔍 Multi-source search (ICD-10-CM, LOINC, RxTerms, HCPCS, UCUM, HPO)

🧠 LLM-based ranking with strict JSON parsing + heuristic fallback

🧩 Grouped output by system and a concise clinical summary

🔁 Automatic retry logic for flaky upstream APIs

⚙️ Environment-based config (TOP_K, OPENAI_MODEL, etc.)

🧱 Built with FastAPI for a clean API + optional web UI

🧱 Project Structure
backend/
  app/
    main.py        # FastAPI app (endpoints, retries, grouped output)
    rank.py        # LLM selection + fallback ranking
    summarize.py   # LLM summary (or fallback)
    models.py      # Pydantic data models
    clients.py     # ICD/LOINC/etc. API clients
web/
  index.html       # optional minimal UI

🚀 Getting Started
1️⃣ Environment Setup
python3.11 -V
python3.11 -m venv .venv311
source .venv311/bin/activate

2️⃣ Install Dependencies
pip install fastapi uvicorn httpx python-dotenv langchain langchain-openai

3️⃣ Configure Environment

Create a .env file in the project root:

OPENAI_API_KEY=sk-your-key
OPENAI_MODEL=gpt-4o-mini
TOP_K=10


💡 The app will still run without an API key (it’ll just use heuristic ranking and a text fallback summary).

⚙️ Run the Server
uvicorn backend.app.main:app --reload --port 8000


Visit http://127.0.0.1:8000/docs

📡 API Overview
POST /search

Request

{ "query": "chest pain" }


Response

{
  "results": [
    { "system": "ICD-10-CM", "code": "R07.9", "display": "Chest pain, unspecified" },
    { "system": "LOINC", "code": "58259-3", "display": "Chest pain" }
  ],
  "summary": "Short plain-English explanation of why these codes match the query.",
  "grouped": {
    "ICD-10-CM": [ { "code": "R07.9", "display": "Chest pain, unspecified" } ],
    "LOINC": [ { "code": "58259-3", "display": "Chest pain" } ]
  }
}


cURL example

curl -s http://127.0.0.1:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query":"chest pain"}' | jq .

GET /health/env

Returns whether your OpenAI key is present and ends with which 6 characters.

🧠 Configuration
Variable	Description	Default
OPENAI_API_KEY	Enables LLM ranking & summary	(none)
OPENAI_MODEL	LLM model used	gpt-4o-mini
TOP_K	Number of top results	10
🧩 How It Works

Fan-out: Queries multiple public vocabularies (Clinical Tables APIs).

Rank: LLM selects top-K codes; if it fails, fallback uses fuzzy match.

Summarize: LLM or fallback summary explains why codes are relevant.

Return: Response includes both flat results and grouped sections.

🧪 Testing

Add simple pytest tests for:

Code normalization (clients)

Ranking deduplication

LLM fallback behavior

/search endpoint returns grouped + summary