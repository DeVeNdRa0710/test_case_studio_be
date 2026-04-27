# ERP Test Case Generator — Backend

Hybrid RAG (Pinecone + Neo4j) + LLM backend that ingests ERP requirements,
Figma JSON, and API specs, then generates structured test cases and
Playwright / Postman automation artifacts.

## Stack
- FastAPI (async)
- Pinecone (vector RAG)
- Neo4j (graph RAG)
- Google Gemini (default LLM + embeddings) — OpenAI / Groq also pluggable
- Pydantic v2 / Loguru / Tenacity

## Provider switching

Set in `.env`:

```env
LLM_PROVIDER=gemini          # gemini | openai | groq
LLM_MODEL=gemini-2.0-flash   # or gemini-2.5-pro for deeper reasoning

EMBEDDINGS_PROVIDER=gemini   # gemini | openai
GEMINI_EMBED_MODEL=text-embedding-004   # 768 dims
PINECONE_DIMENSION=768       # MUST match the embeddings model
```

Switching to OpenAI embeddings (`text-embedding-3-small`) requires
`PINECONE_DIMENSION=1536` and a fresh Pinecone index.

## Project layout
```
app/
  api/            # FastAPI routes
  core/           # config, logging, exceptions
  generators/     # Playwright + Postman emitters
  models/         # internal domain models (graph, chunk)
  prompts/        # LLM prompt templates
  rag/
    graph/        # Neo4j client + store
    vector/       # Pinecone client + store
  schemas/        # request/response pydantic models
  services/       # ingestion, retrieval, testcase, LLM, embeddings
  utils/          # chunker, pdf, json_io
  main.py         # FastAPI entrypoint
```

## Setup
```bash
cp .env.example .env           # fill in keys
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Endpoints
| Method | Path                    | Purpose                                   |
|--------|-------------------------|-------------------------------------------|
| GET    | /health                 | liveness probe                            |
| POST   | /ingest/requirements    | text / PDF / JSON requirements            |
| POST   | /ingest/figma           | parsed Figma JSON                         |
| POST   | /ingest/apis            | OpenAPI-ish spec                          |
| POST   | /generate-testcases     | hybrid-RAG test case generation           |
| POST   | /generate-playwright    | TS Playwright specs from test cases       |
| POST   | /generate-postman       | Postman v2.1 collection from test cases   |

## Examples

### Ingest requirement (JSON body)
```bash
curl -X POST http://localhost:8000/ingest/requirements \
  -H "Content-Type: application/json" \
  -d '{
    "module": "Sales",
    "title": "Sales Order Creation",
    "content": "When a sales order is confirmed, the system reserves inventory, generates an invoice draft, and triggers a shipment request via the Logistics module."
  }'
```

### Ingest requirement (PDF upload)
```bash
curl -X POST http://localhost:8000/ingest/requirements \
  -F module=Sales -F title="SO Spec" -F file=@so_spec.pdf
```

### Generate test cases
```bash
curl -X POST http://localhost:8000/generate-testcases \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Create a sales order and verify downstream invoice + shipment",
    "modules": ["Sales", "Logistics", "Finance"],
    "test_type": "e2e",
    "top_k": 8
  }'
```

Sample response:
```json
{
  "ok": true,
  "retrieved_docs": 6,
  "retrieved_graph_nodes": 11,
  "test_cases": [
    {
      "scenario": "Confirm sales order triggers invoice draft and shipment",
      "modules": ["Sales", "Finance", "Logistics"],
      "preconditions": ["Customer exists", "Item SKU in stock"],
      "steps": [
        {"action": "navigate", "target": "/sales/orders/new", "value": null, "description": "Open new order"},
        {"action": "enter", "target": "#customer", "value": "ACME", "description": "Customer"},
        {"action": "click", "target": "button#confirm", "value": null, "description": "Confirm order"},
        {"action": "validate", "target": ".toast-success", "value": "Order confirmed", "description": "Success toast"}
      ],
      "expected_results": [
        "Invoice draft created in Finance",
        "Shipment request visible in Logistics"
      ],
      "edge_cases": [
        "Confirming with insufficient stock blocks with inventory error"
      ],
      "apis": [
        {"name": "Create Order", "method": "POST", "path": "/api/sales/orders", "body": {"customer": "ACME"}, "expected_status": 201, "save_as": "orderId"},
        {"name": "Fetch Invoice", "method": "GET", "path": "/api/finance/invoices?order={{orderId}}", "expected_status": 200}
      ]
    }
  ]
}
```

### Generate Playwright
```bash
curl -X POST http://localhost:8000/generate-playwright \
  -H "Content-Type: application/json" \
  -d '{ "test_cases": [ ... ], "base_url": "https://erp.example.com" }'
```

### Generate Postman
```bash
curl -X POST http://localhost:8000/generate-postman \
  -H "Content-Type: application/json" \
  -d '{ "test_cases": [ ... ], "collection_name": "ERP — Sales E2E", "base_url": "https://api.erp.example.com" }'
```

## Graph schema
- Nodes: `Module`, `Entity`, `API`, `UIScreen` (unique by `name`)
- Relationships: `CREATES`, `TRIGGERS`, `GENERATES`, `UPDATES`, `DEPENDS_ON`, `CALLS_API`
- Relationship properties: `condition`, `action`, `trigger_point`
