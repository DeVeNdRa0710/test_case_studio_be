from typing import Any

from pydantic import BaseModel, Field


class GenerateTestCasesRequest(BaseModel):
    project: str = Field(
        ...,
        description="Project (tenant) to retrieve context from. Only documents/graph data for this project are used.",
        examples=["project-one"],
    )
    query: str = Field(
        ...,
        description="Natural-language description of the flow you want test cases for.",
        examples=["Create a sales order and verify downstream invoice + shipment"],
    )
    modules: list[str] = Field(
        default_factory=list,
        description="Optional module filter. Restricts Pinecone retrieval and seeds the Neo4j traversal.",
        examples=[["Sales", "Logistics", "Finance"]],
    )
    test_type: str = Field(
        default="integration",
        description="Type of tests to generate: 'integration', 'e2e', or 'api'.",
        examples=["e2e"],
    )
    top_k: int | None = Field(
        default=None,
        description="Override the default number of vector docs to retrieve (defaults to TOP_K from .env).",
        examples=[8],
    )
    extra_context: str | None = Field(
        default=None,
        description="Additional free-form context appended to the LLM prompt (e.g. environment info, custom rules).",
    )


class TestCase(BaseModel):
    scenario: str = Field(..., description="Short, specific name of the scenario being tested.")
    modules: list[str] = Field(default_factory=list, description="Modules involved in this scenario.")
    preconditions: list[str] = Field(default_factory=list, description="State/setup that must hold before the steps run.")
    steps: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Ordered list of step objects: {action, target, value, description}. Supported actions: navigate, enter, click, select, validate, wait, call_api.",
    )
    expected_results: list[str] = Field(default_factory=list, description="Assertions the scenario must satisfy overall.")
    edge_cases: list[str] = Field(default_factory=list, description="Meaningful negative/boundary cases for the same flow.")
    apis: list[dict[str, Any]] = Field(
        default_factory=list,
        description="API calls referenced by this scenario. Each: {name, method, path, body, expected_status, save_as, depends_on}.",
    )


class GenerateTestCasesResponse(BaseModel):
    ok: bool = True
    test_cases: list[TestCase]
    retrieved_docs: int = Field(..., description="How many Pinecone matches fed the LLM prompt.")
    retrieved_graph_nodes: int = Field(..., description="How many Neo4j nodes fed the LLM prompt.")


class GeneratePlaywrightRequest(BaseModel):
    test_cases: list[TestCase] = Field(
        ...,
        description="Test cases to convert into Playwright spec files (typically the output of /generate-testcases).",
    )
    base_url: str | None = Field(
        default=None,
        description="Base URL prefixed to relative targets when no explicit 'navigate' step exists.",
        examples=["https://erp.example.com"],
    )
    language: str = Field(
        default="typescript",
        description="Language for emitted Playwright files: 'typescript' or 'javascript'.",
    )


class GeneratePlaywrightResponse(BaseModel):
    ok: bool = True
    files: list[dict[str, str]] = Field(
        default_factory=list,
        description="Emitted files as [{filename, content}]. Write each to disk and run `npx playwright test`.",
    )


class GeneratePostmanRequest(BaseModel):
    test_cases: list[TestCase] = Field(
        ...,
        description="Test cases whose `apis` arrays will be converted into Postman v2.1 folders + requests.",
    )
    collection_name: str = Field(default="ERP Generated Collection", description="Name shown in Postman after import.")
    base_url: str | None = Field(
        default=None,
        description="Base URL used to construct absolute request URLs and the `{{baseUrl}}` collection variable.",
        examples=["https://api.erp.example.com"],
    )


class GeneratePostmanResponse(BaseModel):
    ok: bool = True
    collection: dict[str, Any] = Field(..., description="Postman Collection v2.1 JSON. Import directly into Postman or run via Newman.")
