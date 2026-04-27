from fastapi import APIRouter

from app.generators.playwright_generator import get_playwright_generator
from app.generators.postman_generator import get_postman_generator
from app.schemas.testcase import (
    GeneratePlaywrightRequest,
    GeneratePlaywrightResponse,
    GeneratePostmanRequest,
    GeneratePostmanResponse,
    GenerateTestCasesRequest,
    GenerateTestCasesResponse,
)
from app.services.project_service import get_project_service
from app.services.testcase_service import get_testcase_service

router = APIRouter(tags=["generate"])


@router.post(
    "/generate-testcases",
    response_model=GenerateTestCasesResponse,
    summary="Generate structured test cases from a query",
    description=(
        "Hybrid RAG pipeline:\n"
        "1. Embeds `query` and retrieves top-K chunks from Pinecone (filtered by `modules`).\n"
        "2. Queries Neo4j for dependency paths (2-hop) seeded from `modules` + query keywords.\n"
        "3. Merges both contexts and calls the LLM (JSON mode) to emit structured test cases."
    ),
)
async def generate_testcases(payload: GenerateTestCasesRequest) -> GenerateTestCasesResponse:
    """
    Field-by-field example:

    - **query** — what flow to generate tests for (plain English).
      Example: `"Create a sales order and verify downstream invoice + shipment"`

    - **modules** — scope filter for both Pinecone + Neo4j.
      Example: `["Sales", "Logistics", "Finance"]`  (empty list = all modules)

    - **test_type** — `"integration"`, `"e2e"`, or `"api"`. Injected into the prompt.
      Example: `"e2e"`

    - **top_k** — override `.env` `TOP_K`. Higher = more context, slower, costlier.
      Example: `8`

    - **extra_context** — optional free-form text appended to the prompt.
      Example: `"Target env is staging. Tax rules: GST 18%. Assume seeded customer 'ACME'."`

    Full payload example:
    ```json
    {
      "query": "Create a sales order and verify downstream invoice + shipment",
      "modules": ["Sales", "Logistics", "Finance"],
      "test_type": "e2e",
      "top_k": 8
    }
    ```
    """
    payload.project = get_project_service().require(payload.project)
    svc = get_testcase_service()
    cases, docs, nodes = await svc.generate(payload)
    return GenerateTestCasesResponse(
        test_cases=cases, retrieved_docs=docs, retrieved_graph_nodes=nodes
    )


@router.post(
    "/generate-playwright",
    response_model=GeneratePlaywrightResponse,
    summary="Emit Playwright spec files from test cases",
    description=(
        "Deterministic (no-LLM) emitter. Each test case → one `.spec.ts` file.\n\n"
        "Action mapping:\n"
        "- navigate → page.goto\n"
        "- enter    → page.fill\n"
        "- click    → page.click\n"
        "- select   → page.selectOption\n"
        "- validate → expect(...).toBeVisible() / toHaveText()\n"
        "- wait     → page.waitForTimeout\n"
        "- call_api → request.<method>"
    ),
)
async def generate_playwright(payload: GeneratePlaywrightRequest) -> GeneratePlaywrightResponse:
    """
    Field-by-field example:

    - **test_cases** — usually the `test_cases` array returned from
      `/generate-testcases`. Each case's `steps[]` is emitted into a Playwright file.
      Step-to-Playwright mapping:
      ```
      {"action":"navigate", "target":"/sales/orders/new"}     -> await page.goto('/sales/orders/new')
      {"action":"enter",    "target":"#customer", "value":"ACME"} -> await page.fill('#customer', 'ACME')
      {"action":"click",    "target":"button#confirm"}        -> await page.click('button#confirm')
      {"action":"validate", "target":".toast-success", "value":"Order confirmed"}
                                                              -> await expect(page.locator('.toast-success')).toHaveText('Order confirmed')
      ```

    - **base_url** — used only when a test case has no explicit `navigate` step.
      Example: `"https://erp.example.com"`

    - **language** — `"typescript"` (default) or `"javascript"`.

    Full payload example:
    ```json
    {
      "test_cases": [ /* output from /generate-testcases */ ],
      "base_url": "https://erp.example.com",
      "language": "typescript"
    }
    ```
    """
    gen = get_playwright_generator()
    files = gen.generate(payload.test_cases, base_url=payload.base_url)
    return GeneratePlaywrightResponse(files=files)


@router.post(
    "/generate-postman",
    response_model=GeneratePostmanResponse,
    summary="Emit a Postman v2.1 collection from test cases",
    description=(
        "Builds a Postman Collection (v2.1). One folder per scenario, one request per API.\n\n"
        "Chaining: when an API has `save_as`, a test script stores the response id in "
        "`pm.collectionVariables` so subsequent requests can reference it via `{{name}}`."
    ),
)
async def generate_postman(payload: GeneratePostmanRequest) -> GeneratePostmanResponse:
    """
    Field-by-field example:

    - **test_cases** — each case becomes a folder; each entry in its `apis[]`
      becomes a request. Request chaining is driven by `save_as` / `depends_on`.
      Example `apis` entry:
      ```json
      {
        "name": "Create Order",
        "method": "POST",
        "path": "/api/sales/orders",
        "body": {"customer": "ACME"},
        "expected_status": 201,
        "save_as": "orderId"
      }
      ```
      A later request can reference it:
      `"path": "/api/finance/invoices?order={{orderId}}"`

    - **collection_name** — name shown in Postman after import.
      Example: `"ERP — Sales E2E"`

    - **base_url** — prefixes every request URL and sets the `{{baseUrl}}` variable.
      Example: `"https://api.erp.example.com"`

    Full payload example:
    ```json
    {
      "test_cases": [ /* output from /generate-testcases */ ],
      "collection_name": "ERP — Sales E2E",
      "base_url": "https://api.erp.example.com"
    }
    ```
    """
    gen = get_postman_generator()
    collection = gen.generate(
        payload.test_cases,
        collection_name=payload.collection_name,
        base_url=payload.base_url,
    )
    return GeneratePostmanResponse(collection=collection)
