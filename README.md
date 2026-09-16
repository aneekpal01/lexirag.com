# LexiRAG: Enterprise Legal Research RAG Backend

[![FastAPI](https://img.shields.io/badge/FastAPI-0.115.6-009688.svg?style=flat&logo=fastapi)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2.60-blue.svg?style=flat)](https://github.com/langchain-ai/langgraph)
[![Nebius Token Factory](https://img.shields.io/badge/Nebius_Token_Factory-NVIDIA_Nemotron-76B900.svg?style=flat)](https://api.tokenfactory.nebius.com)
[![Vector DB](https://img.shields.io/badge/Qdrant_Cloud-BGE--M3-red.svg?style=flat&logo=qdrant)](https://qdrant.tech)
[![Auth](https://img.shields.io/badge/Auth-Clerk_JWT-6C47FF.svg?style=flat)](https://clerk.com)

**LexiRAG** is a production-grade legal research RAG SaaS backend engineered specifically for the Indian legal and regulatory ecosystem. It serves Indian advocates, corporate legal teams, Chartered Accountants (CAs), HR compliance officers, and startup founders navigating the labyrinth of Indian codified law:

- **Companies Act, 2013** & Ministry of Corporate Affairs (MCA) notifications
- **Income Tax Act, 1961** & Central Board of Direct Taxes (CBDT) circulars
- **Goods and Services Tax (GST) Acts, 2017**
- **Insolvency and Bankruptcy Code (IBC), 2016**
- **Bharatiya Nyaya Sanhita (BNS), 2023** & procedural codes
- **Labour Codes & FEMA Regulations**

LexiRAG enforces zero-hallucination statutory grounding by synthesizing dense vector search with multi-hop legal reasoning powered by **NVIDIA Nemotron** models running on **Nebius Token Factory**.

---

## Architecture & LangGraph Pipeline

```
                     ┌────────────────────────────────┐
                     │ Client Request: POST /query    │
                     └───────────────┬────────────────┘
                                     │
                                     ▼
                     ┌────────────────────────────────┐
                     │ Clerk JWT Authentication Guard │
                     └───────────────┬────────────────┘
                                     │
                                     ▼
                     ┌────────────────────────────────┐
                     │  1. Query Classification Node  │
                     │  (nvidia/nemotron-3-nano-30b)  │
                     └───────────────┬────────────────┘
                                     │
                                     ▼
                     ┌────────────────────────────────┐
                     │   2. Statutory Retrieval Node  │
                     │   - Embed: BAAI/bge-m3 via     │
                     │     Nebius Token Factory       │
                     │   - Vector DB: Qdrant Cloud    │
                     └───────────────┬────────────────┘
                                     │
                      [Complexity Routing Decision]
                                     ├───────────────────────────────┐
                                     ▼ (Simple)                      ▼ (Complex)
                     ┌───────────────────────────────┐ ┌───────────────────────────────┐
                     │  3a. Statutory Synthesis Node │ │ 3b. Legal Multi-Hop Reasoning │
                     │  nvidia/nemotron-3-nano-30b   │ │ nvidia/nemotron-3-super-120b  │
                     └───────────────┬───────────────┘ └───────────────┬───────────────┘
                                     │                                 │
                                     └───────────────┬─────────────────┘
                                                     │
                                                     ▼
                                     ┌────────────────────────────────┐
                                     │  4. Citation Formatting Node   │
                                     │  (Deduplication & Ratio Dec.)  │
                                     └───────────────┬────────────────┘
                                                     │
                                                     ▼
                                     ┌────────────────────────────────┐
                                     │ Structured Response + Latency  │
                                     └────────────────────────────────┘
```

### Dynamic Model Routing Strategy
- **`nvidia/nemotron-3-nano-30b-a3b` (Fast / Low Latency)**: Handles single-section threshold checks, filing timelines, penalty formulas, and straightforward definition lookups (e.g., *"What is the statutory threshold for mandatory internal audit under Section 138 of Companies Act 2013?"*).
- **`nvidia/nemotron-3-super-120b-a12b` (Deep Multi-Hop Legal Reasoning)**: Activated when a query spans overlapping statutes, conflicting ratios from High Courts or the Supreme Court, or corporate reorganizations (e.g., *"How does the moratorium under Section 14 IBC override provisional attachment orders passed under Section 5 PMLA?"*).

---

## Nebius Token Factory & NVIDIA Nemotron Integration

The entire inference and vector embedding pipeline runs strictly through **Nebius Token Factory** (`https://api.tokenfactory.nebius.com/v1/`).

### CRITICAL INTEGRATION DETAIL: `reasoning_content` Parsing
NVIDIA Nemotron models deployed on Nebius Token Factory operate as frontier reasoning architectures. Consequently, **the model's generated legal rationale and conclusion are returned in the `reasoning_content` field of the completion choice, NOT inside the standard OpenAI `content` field**.

Standard code assuming `choice.message.content` will encounter empty strings or `None`.

LexiRAG implements a defensive extractor ([`app/clients/nebius.py`](file:///app/clients/nebius.py)):

```python
def extract_reasoning_or_content(choice_message: Any) -> str:
    # 1. Check primary reasoning_content attribute
    reasoning = getattr(choice_message, "reasoning_content", None)
    if reasoning and str(reasoning).strip():
        return str(reasoning).strip()

    # 2. Check Pydantic v2 dynamic model_extra container
    if hasattr(choice_message, "model_extra") and choice_message.model_extra:
        extra_reasoning = choice_message.model_extra.get("reasoning_content")
        if extra_reasoning and str(extra_reasoning).strip():
            return str(extra_reasoning).strip()

    # 3. Check dictionary representation
    if isinstance(choice_message, dict):
        dict_reasoning = choice_message.get("reasoning_content")
        if dict_reasoning and str(dict_reasoning).strip():
            return str(dict_reasoning).strip()

    # 4. Graceful fallback to standard content
    content = getattr(choice_message, "content", None)
    if content and str(content).strip():
        return str(content).strip()

    raise NebiusExtractionError("Nebius response contained neither reasoning_content nor content")
```

---

## Project Structure

```
lexirag-backend/
├── app/
│   ├── api/
│   │   ├── dependencies.py       # FastAPI dependency injection (Nebius, Qdrant, Graph)
│   │   └── routes.py             # Route controllers (POST /query, GET /health)
│   ├── auth/
│   │   └── clerk.py              # Clerk JWT verification dependency with dev mode bypass
│   ├── clients/
│   │   ├── nebius.py             # Nebius Token Factory client with retry & reasoning parsing
│   │   └── qdrant.py             # Qdrant Cloud client wrapper with defensive payload parsing
│   ├── core/
│   │   ├── config.py             # Pydantic Settings with env parsing & validations
│   │   ├── constants.py          # Domain constants, timeouts, top-k limits
│   │   ├── exceptions.py         # Domain exception hierarchy (429, 502, 503, 401)
│   │   └── logging.py            # Structured logging setup
│   ├── rag/
│   │   ├── graph.py              # Compiled LangGraph StateGraph builder
│   │   ├── nodes.py              # The 4 discrete execution nodes
│   │   └── state.py              # TypedDict LegalGraphState definition
│   ├── schemas/
│   │   ├── errors.py             # Standardized error response models
│   │   └── query.py              # Request, response, and citation schemas
│   └── main.py                   # FastAPI app factory, lifespan, and exception handlers
├── tests/
│   ├── test_api.py               # Integration tests for endpoints, auth, and error codes
│   ├── test_nebius_extraction.py # Tests verifying reasoning_content extraction
│   └── test_pipeline.py          # Tests for LangGraph nodes and end-to-end flow
├── .env.example                  # Environment configuration template
├── requirements.txt              # Exact pinned dependencies
├── README.md                     # Documentation for judges and developers
└── NEBIUS_FEEDBACK.md            # Actionable feedback on Nebius Token Factory & NVIDIA tools
```

---

## Setup & Local Run Instructions

### Prerequisites
- Python 3.11, 3.12, or 3.13
- A Nebius Token Factory API Key ([Nebius Studio](https://tokenfactory.nebius.com))
- A Qdrant Cloud cluster or local instance with pre-embedded legal documents

### 1. Clone & Environment Setup
```bash
# Navigate to the project root
cd lexirag-backend

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# Install exact pinned dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env` and fill in your credentials:
```bash
cp .env.example .env
```

Key variables to configure:
```ini
NEBIUS_API_KEY=your_nebius_token_factory_key
NEBIUS_BASE_URL=https://api.tokenfactory.nebius.com/v1/
NEMOTRON_NANO_MODEL=nvidia/nemotron-3-nano-30b-a3b
NEMOTRON_SUPER_MODEL=nvidia/nemotron-3-super-120b-a12b
EMBEDDING_MODEL=BAAI/bge-m3
QDRANT_URL=https://your-cluster-id.us-east-1-0.aws.cloud.qdrant.io:6333
QDRANT_API_KEY=your_qdrant_api_key
CLERK_DEV_MODE=true
```

### 3. Run the Backend Service
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The interactive OpenAPI Swagger UI is available at:
`http://localhost:8000/docs`

---

## Running the Automated Test Suite

The test suite validates:
1. `reasoning_content` extraction from mock Nebius responses (attribute, dict, and model_extra formats).
2. LangGraph state transitions across all 4 nodes.
3. FastAPI endpoint status codes: 200 (Success), 401 (Unauthorized), 422 (Validation), 429 (Nebius Rate Limit with Retry-After header), and 503 (Qdrant Cloud Unavailable).

```bash
pytest -v
```

---

## API Usage & cURL Examples

### 1. Health Probe
```bash
curl -X GET "http://localhost:8000/api/v1/health" \
     -H "Accept: application/json"
```

### 2. Simple Legal Query (Routes to Nemotron Nano)
```bash
curl -X POST "http://localhost:8000/query" \
     -H "Authorization: Bearer dev-test-token" \
     -H "Content-Type: application/json" \
     -d '{
       "query": "What are the statutory conditions under Section 185 of the Companies Act 2013 for advancing loans to a private company having common directors?",
       "domain": "corporate_law",
       "jurisdiction": "India"
     }'
```

**Example Response:**
```json
{
  "answer": "Under Section 185 of the Companies Act, 2013, a company may advance any loan including any loan represented by a book debt, or give any guarantee or provide any security in connection with any loan taken by any private company in which any director is interested, subject to the condition that a special resolution is passed by the company in general meeting...",
  "citations": [
    {
      "act_name": "Companies Act, 2013",
      "section": "Section 185",
      "sub_section": "(2)",
      "title": "Loans to directors, etc.",
      "court_or_authority": null,
      "citation_ref": null,
      "relevance_excerpt": "A company may advance any loan including any loan represented by a book debt...",
      "similarity_score": 0.9124
    }
  ],
  "complexity": "simple",
  "model_used": "nvidia/nemotron-3-nano-30b-a3b",
  "retrieved_chunks_count": 1,
  "execution_time_ms": 782.45,
  "fallback_triggered": false
}
```

### 3. Complex Legal Query (Routes to Nemotron Super 120b)
```bash
curl -X POST "http://localhost:8000/query" \
     -H "Authorization: Bearer dev-test-token" \
     -H "Content-Type: application/json" \
     -d '{
       "query": "How does the moratorium declared under Section 14 of the Insolvency and Bankruptcy Code 2016 impact provisional attachment orders previously passed by the Enforcement Directorate under Section 5 of the Prevention of Money Laundering Act 2002? Analyze the ratio laid down by the Delhi High Court in Rajiv Chakraborty vs ED.",
       "domain": "corporate_law",
       "jurisdiction": "India"
     }'
```

---

## Error Handling Specifications

LexiRAG adheres to strict production resilience patterns:
- **HTTP 401 Unauthorized**: Missing or malformed Clerk JWT token.
- **HTTP 422 Unprocessable Entity**: Queries shorter than 5 characters or exceeding 2000 characters.
- **HTTP 429 Too Many Requests**: Triggered on Nebius Token Factory rate limits. Returns structured error body along with `Retry-After: <seconds>` HTTP header.
- **HTTP 502 Bad Gateway**: Raised when Nebius Token Factory encounters an upstream network failure or returns a payload lacking both `reasoning_content` and `content`.
- **HTTP 503 Service Unavailable**: Raised when Qdrant Cloud cluster cannot be reached.
- **HTTP 504 Gateway Timeout**: Raised if an inference request exceeds `NEBIUS_REQUEST_TIMEOUT_SECONDS` (60s).
