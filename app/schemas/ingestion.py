from typing import Any

from pydantic import BaseModel, Field


class RequirementIngestRequest(BaseModel):
    project: str = Field(
        ...,
        description="Project (tenant) this document belongs to. Used as the Pinecone namespace and the `project` property on every Neo4j node.",
        examples=["project-one"],
    )
    module: str = Field(
        ...,
        description="ERP module this requirement belongs to. Becomes a Module node in Neo4j and a metadata filter in Pinecone. Example: 'Sales', 'Inventory', 'Finance'.",
        examples=["Sales"],
    )
    title: str = Field(
        ...,
        description="Human-friendly document title, stored as metadata for traceability.",
        examples=["Sales Order Creation Spec v2"],
    )
    content: str = Field(
        ...,
        description="Raw requirement text. It is chunked, embedded, and pushed to Pinecone + Neo4j.",
        examples=[
            "When a sales order is confirmed, reserve inventory, generate an invoice draft, and trigger a shipment request."
        ],
    )
    source: str | None = Field(
        default=None,
        description="Optional provenance tag — where this doc came from. Stored as metadata only. Example: 'confluence', 'jira:ERP-123', 'manual'.",
        examples=["confluence"],
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Freeform extra metadata (version, author, tags) stored alongside the embedding.",
        examples=[{"version": "v2", "owner": "pm@acme.com"}],
    )


class FigmaIngestRequest(BaseModel):
    project: str = Field(
        ...,
        description="Project (tenant) this screen belongs to.",
        examples=["project-one"],
    )
    module: str = Field(
        ...,
        description="ERP module this screen belongs to. Creates a (:UIScreen)-[:DEPENDS_ON]->(:Module) edge.",
        examples=["Sales"],
    )
    screen_name: str = Field(
        ...,
        description="Canonical screen name. Becomes a unique :UIScreen node in Neo4j.",
        examples=["NewOrderScreen"],
    )
    figma_json: dict[str, Any] = Field(
        ...,
        description="Preprocessed Figma export. Nested tree of frames/components; the service walks name/type/children.",
        examples=[
            {
                "name": "NewOrder",
                "type": "FRAME",
                "children": [
                    {"type": "INPUT", "name": "customer"},
                    {"type": "BUTTON", "name": "confirm"},
                ],
            }
        ],
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Freeform extra metadata (figma file id, version, designer, etc.).",
        examples=[{"figma_file": "abc123", "version": "v3"}],
    )


class ApiSpecIngestRequest(BaseModel):
    project: str = Field(
        ...,
        description="Project (tenant) this API spec belongs to.",
        examples=["project-one"],
    )
    module: str = Field(
        ...,
        description="ERP module this API spec belongs to. All extracted endpoints become :API nodes linked to this Module.",
        examples=["Sales"],
    )
    spec: dict[str, Any] = Field(
        ...,
        description="OpenAPI/Swagger-like JSON. The service iterates spec.paths.* to create API nodes.",
        examples=[
            {
                "paths": {
                    "/api/sales/orders": {
                        "post": {"summary": "Create order", "operationId": "createOrder"}
                    }
                }
            }
        ],
    )
    description: str | None = Field(
        default=None,
        description="Optional description of the API surface. Embedded for retrieval.",
        examples=["Sales public REST API v1"],
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Freeform extra metadata (spec version, owning team, base path, etc.).",
        examples=[{"version": "1.4", "owner": "sales-platform"}],
    )


class IngestResponse(BaseModel):
    ok: bool = True
    doc_id: str = Field(..., description="Generated identifier used as a key in Pinecone.")
    chunks_indexed: int = Field(..., description="Number of vector chunks written to Pinecone.")
    nodes_upserted: int = Field(..., description="Number of graph nodes MERGEd into Neo4j.")
    relationships_upserted: int = Field(..., description="Number of graph relationships MERGEd into Neo4j.")


class BatchDocResult(BaseModel):
    ok: bool = Field(..., description="True if this file was ingested successfully.")
    file_name: str = Field(..., description="Original uploaded filename.")
    title: str = Field(..., description="Title used for this document (filename-derived or prefixed).")
    doc_id: str | None = Field(default=None, description="Generated doc_id if successful.")
    chunks_indexed: int = 0
    nodes_upserted: int = 0
    relationships_upserted: int = 0
    error: str | None = Field(default=None, description="Error message if this file failed.")


class BatchTotals(BaseModel):
    chunks_indexed: int = 0
    nodes_upserted: int = 0
    relationships_upserted: int = 0


class BatchIngestResponse(BaseModel):
    ok: bool = True
    total_files: int
    succeeded: int
    failed: int
    results: list[BatchDocResult]
    totals: BatchTotals


class FigmaBatchResult(BaseModel):
    ok: bool = Field(..., description="True if this image was extracted (and ingested, if auto_ingest=true).")
    file_name: str = Field(..., description="Original uploaded filename.")
    screen_name: str = Field(..., description="Screen name used for this image.")
    module: str = Field(..., description="Module this screen was attached to.")
    figma_json: dict[str, Any] | None = Field(
        default=None,
        description="Extracted UI tree. Present on success, null on failure.",
    )
    ingested: bool = Field(default=False, description="True if the screen was written to Pinecone + Neo4j.")
    doc_id: str | None = Field(default=None, description="Pinecone doc id when ingested.")
    chunks_indexed: int = 0
    nodes_upserted: int = 0
    relationships_upserted: int = 0
    error: str | None = Field(default=None, description="Error message when this image failed.")


class FigmaBatchResponse(BaseModel):
    ok: bool = True
    total_files: int
    succeeded: int
    failed: int
    results: list[FigmaBatchResult]
    totals: BatchTotals
