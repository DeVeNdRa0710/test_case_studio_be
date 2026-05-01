import json
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel

from app.core.config import settings
from app.core.exceptions import IngestionError
from app.core.logging import logger
from app.core.rate_limit import limiter
from app.schemas.ingestion import (
    ApiSpecIngestRequest,
    BatchDocResult,
    BatchIngestResponse,
    BatchTotals,
    FigmaFromUrlResponse,
    FigmaIngestRequest,
    FigmaUrlIngestRequest,
    IngestResponse,
    RequirementIngestRequest,
)
from app.services.api_normalizer import (
    SUPPORTED_FRAMEWORKS,
    from_image as normalize_image,
    from_openapi,
    from_postman,
    from_routes,
    from_text,
)
from app.services.figma_api import (
    FigmaAuthError,
    FigmaNotFoundError,
    figma_url_to_tree,
)
from app.services.ingestion_service import get_ingestion_service
from app.services.project_service import get_project_service
from app.services.vision import LLMModelNotFoundError, LLMQuotaError
from app.utils.pdf import extract_text_from_pdf
from app.utils.uploads import assert_pdf_page_limit, read_and_validate

router = APIRouter(prefix="/ingest", tags=["ingest"])


class NormalizeApiSpecResponse(BaseModel):
    ok: bool = True
    spec: dict[str, Any]
    paths_count: int
    endpoints_count: int
    format: str


@router.post(
    "/requirements",
    response_model=IngestResponse,
    summary="Ingest a requirements document",
    description=(
        "Upload a requirements document into the hybrid RAG store. You can either:\n"
        "- POST JSON (use the `body` field) matching `RequirementIngestRequest`, or\n"
        "- POST multipart/form-data with `module`, `title`, and **one of** `text` or `file` (PDF/TXT).\n\n"
        "The content is chunked + embedded into Pinecone and an LLM extracts a graph of "
        "Module/Entity/API/UIScreen nodes and their relationships into Neo4j."
    ),
)
@limiter.limit(settings.rate_limit_ingest)
async def ingest_requirements(
    request: Request,
    project: str | None = Form(
        default=None,
        description="Project (tenant) name. Pinecone namespace + Neo4j node scope.",
    ),
    module: str | None = Form(
        default=None,
        description="ERP module name (e.g. 'Sales'). Used as vector metadata and graph node.",
    ),
    title: str | None = Form(
        default=None,
        description="Human-friendly document title.",
    ),
    source: str | None = Form(
        default=None,
        description="Optional provenance tag (e.g. 'confluence', 'jira:ERP-123').",
    ),
    text: str | None = Form(
        default=None,
        description="Raw requirement text. Use this when not uploading a file.",
    ),
    file: UploadFile | None = File(
        default=None,
        description="Alternative to `text`: upload a .pdf or .txt file. PDFs are parsed via pypdf.",
    ),
    body: RequirementIngestRequest | None = None,
) -> IngestResponse:
    """
    Field-by-field example:

    - **module** — which ERP area this doc belongs to. Becomes a :Module node
      and a Pinecone metadata filter.
      Example: `"Sales"`  (others: `"Inventory"`, `"Finance"`, `"HR"`)

    - **title** — short human-readable name. Stored as metadata for traceability.
      Example: `"Sales Order Creation Spec v2"`

    - **source** — optional provenance. Stored as metadata only.
      Example: `"confluence"`, `"jira:ERP-123"`, `"manual"`, `None`

    - **text** — paste the full requirement. Use this OR `file`, not both.
      Example:
      `"When a sales order is confirmed, reserve inventory, generate an invoice draft, and trigger a shipment request."`

    - **file** — upload a `.pdf` or `.txt`. Use this OR `text`, not both.
      PDFs are parsed with pypdf. Example: `so_spec.pdf` (binary upload)

    - **body** — full JSON alternative to the form fields.
      Example:
      ```json
      {
        "module": "Sales",
        "title": "Sales Order Creation Spec v2",
        "content": "When a sales order is confirmed, reserve inventory ...",
        "source": "confluence",
        "metadata": {"version": "v2"}
      }
      ```

    End-to-end curl example (form-data with pasted text):
    ```
    curl -X POST http://localhost:8080/ingest/requirements \\
      -F module=Sales -F title="SO Flow" \\
      -F text="When a sales order is confirmed, reserve inventory..."
    ```
    """
    svc = get_ingestion_service()

    if body is not None:
        body.project = await get_project_service().require_async(body.project)
        doc_id, indexed, nodes, rels = await svc.ingest_requirement(body)
        return IngestResponse(
            doc_id=doc_id, chunks_indexed=indexed, nodes_upserted=nodes, relationships_upserted=rels
        )

    if not project or not title:
        raise HTTPException(
            status_code=422, detail="`project` and `title` are required"
        )
    project = await get_project_service().require_async(project)

    content = text or ""
    if file is not None:
        raw, mime = await read_and_validate(file, kinds={"pdf", "text"})
        if mime == "application/pdf":
            assert_pdf_page_limit(raw)
            content = extract_text_from_pdf(raw)
        else:
            content = raw.decode("utf-8", errors="ignore")

    if not content.strip():
        raise IngestionError("No requirement content provided")

    doc_id, indexed, nodes, rels = await svc.ingest_requirement_text(
        project=project, module=module, title=title, content=content, source=source
    )
    logger.info(f"Ingested requirement {doc_id}: chunks={indexed}, nodes={nodes}, rels={rels}")
    return IngestResponse(
        doc_id=doc_id, chunks_indexed=indexed, nodes_upserted=nodes, relationships_upserted=rels
    )


@router.post(
    "/requirements/batch",
    response_model=BatchIngestResponse,
    summary="Ingest multiple requirement files in one request",
    description=(
        "Upload 1..N PDF/TXT files. Each file is ingested as its own document:\n"
        "- Title defaults to the filename (extension stripped).\n"
        "- If `title_prefix` is provided, the per-doc title becomes `'<prefix> — <filename>'`.\n"
        "- Files are processed sequentially. A failure on one file does not abort "
        "  the batch — failed files are reported inline with `ok=false` and an `error` message.\n"
        "- `module` and `source` apply to every file."
    ),
)
@limiter.limit(settings.rate_limit_ingest)
async def ingest_requirements_batch(
    request: Request,
    files: list[UploadFile] = File(
        ...,
        description="One or more .pdf or .txt files. Use the same field name 'files' for each.",
    ),
    project: str = Form(..., description="Project (tenant) name applied to every file."),
    module: str | None = Form(
        default=None,
        description="Optional ERP module applied to every file.",
    ),
    title_prefix: str | None = Form(
        default=None,
        description="Optional prefix prepended to each file's filename-derived title.",
    ),
    source: str | None = Form(
        default=None,
        description="Optional provenance tag applied to every file.",
    ),
) -> BatchIngestResponse:
    if not files:
        raise HTTPException(status_code=422, detail="At least one file is required")
    if not project.strip():
        raise HTTPException(status_code=422, detail="`project` is required")
    project = await get_project_service().require_async(project)

    svc = get_ingestion_service()
    results: list[BatchDocResult] = []
    totals = BatchTotals()
    prefix = (title_prefix or "").strip()
    module_clean = (module or "").strip() or None

    for f in files:
        file_name = f.filename or "upload"
        base = file_name.rsplit(".", 1)[0] if "." in file_name else file_name
        title = f"{prefix} — {base}" if prefix else base

        try:
            raw, mime = await read_and_validate(f, kinds={"pdf", "text"})
            if mime == "application/pdf":
                assert_pdf_page_limit(raw)
                content = extract_text_from_pdf(raw)
            else:
                content = raw.decode("utf-8", errors="ignore")
            if not content.strip():
                raise IngestionError("No extractable text")

            doc_id, indexed, nodes, rels = await svc.ingest_requirement_text(
                project=project,
                module=module_clean,
                title=title,
                content=content,
                source=(source or "").strip() or None,
            )
            totals.chunks_indexed += indexed
            totals.nodes_upserted += nodes
            totals.relationships_upserted += rels
            results.append(
                BatchDocResult(
                    ok=True,
                    file_name=file_name,
                    title=title,
                    doc_id=doc_id,
                    chunks_indexed=indexed,
                    nodes_upserted=nodes,
                    relationships_upserted=rels,
                )
            )
            logger.info(
                f"Batch ingested {doc_id} ({file_name}): "
                f"chunks={indexed}, nodes={nodes}, rels={rels}"
            )
        except Exception as exc:
            logger.error(f"Batch ingest failed for {file_name}: {exc}")
            results.append(
                BatchDocResult(
                    ok=False,
                    file_name=file_name,
                    title=title,
                    error=str(exc),
                )
            )

    succeeded = sum(1 for r in results if r.ok)
    return BatchIngestResponse(
        ok=succeeded > 0,
        total_files=len(files),
        succeeded=succeeded,
        failed=len(files) - succeeded,
        results=results,
        totals=totals,
    )


@router.post(
    "/figma/from-url",
    response_model=FigmaFromUrlResponse,
    summary="Fetch a Figma design directly from its URL via Figma's REST API",
    description=(
        "Pulls the actual Figma node tree using the Figma REST API — no screenshots, "
        "no preprocessed JSON. Provide a `figma_url` and an access token (in the body "
        "or via the server's `FIGMA_TOKEN` env var).\n\n"
        "Supported URL shapes:\n"
        "- `https://www.figma.com/design/<fileKey>/<name>?node-id=<id>` (preferred — fetches just that frame)\n"
        "- `https://www.figma.com/design/<fileKey>/<name>` (no node-id — fetches the whole document)\n"
        "- `https://www.figma.com/file/<fileKey>/...` (legacy)\n"
        "- branch URLs `/design/<fileKey>/branch/<branchKey>/<name>`\n\n"
        "When `auto_ingest=true` (default), the fetched tree is written to Pinecone + Neo4j "
        "as a `:UIScreen`. When false, it's returned for review only."
    ),
)
@limiter.limit(settings.rate_limit_ingest)
async def ingest_figma_from_url(
    request: Request, payload: FigmaUrlIngestRequest
) -> FigmaFromUrlResponse:
    payload.project = await get_project_service().require_async(payload.project)

    # Token is optional — only protected Figma files need it. We pass through
    # whatever's provided (request → env fallback → none) and let the Figma API
    # tell us if auth is actually required.
    token = (payload.figma_token or "").strip() or settings.figma_token

    try:
        fetched = await figma_url_to_tree(
            figma_url=payload.figma_url,
            token=token,
            screen_name=payload.screen_name,
            module=payload.module,
        )
    except FigmaAuthError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    except FigmaNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    response = FigmaFromUrlResponse(
        figma_json=fetched.tree,
        screen_name=fetched.derived_screen_name,
        module=fetched.derived_module,
        file_key=fetched.parts.file_key,
        node_id=fetched.parts.node_id,
    )

    if not payload.auto_ingest:
        return response

    svc = get_ingestion_service()
    figma_metadata = {
        "source": "figma_url",
        "figma_file_key": fetched.parts.file_key,
        "figma_node_id": fetched.parts.node_id,
        "figma_file_name": fetched.file_name,
        "figma_url": payload.figma_url,
        **payload.metadata,
    }
    doc_id, indexed, nodes, rels = await svc.ingest_figma(
        FigmaIngestRequest(
            project=payload.project,
            module=fetched.derived_module,
            screen_name=fetched.derived_screen_name,
            figma_json=fetched.tree,
            metadata=figma_metadata,
        )
    )
    response.ingested = True
    response.doc_id = doc_id
    response.chunks_indexed = indexed
    response.nodes_upserted = nodes
    response.relationships_upserted = rels
    logger.info(
        f"Figma URL ingested: project={payload.project} "
        f"screen={fetched.derived_screen_name} module={fetched.derived_module} "
        f"file={fetched.parts.file_key} node={fetched.parts.node_id} doc={doc_id}"
    )
    return response


@router.post(
    "/apis",
    response_model=IngestResponse,
    summary="Ingest an API specification",
    description=(
        "Register an OpenAPI-style spec. The service:\n"
        "- Walks `spec.paths.*` and creates one :API node per endpoint (name = 'METHOD /path').\n"
        "- Links each API to the provided Module via :DEPENDS_ON.\n"
        "- Embeds a flattened textual summary of the spec into Pinecone for retrieval."
    ),
)
@limiter.limit(settings.rate_limit_ingest)
async def ingest_apis(request: Request, payload: ApiSpecIngestRequest) -> IngestResponse:
    """
    Field-by-field example:

    - **module** — ERP module that owns these endpoints.
      Example: `"Sales"`

    - **spec** — OpenAPI-ish JSON. Only `spec.paths.*` is required; other
      fields (info, components, …) are ignored but preserved in embeddings.
      Example:
      ```json
      {
        "paths": {
          "/api/sales/orders": {
            "post": {"summary": "Create order", "operationId": "createOrder"},
            "get":  {"summary": "List orders",  "operationId": "listOrders"}
          },
          "/api/sales/orders/{id}/confirm": {
            "post": {"summary": "Confirm order", "operationId": "confirmOrder"}
          }
        }
      }
      ```
      → Creates `:API` nodes named `"POST /api/sales/orders"`,
        `"GET /api/sales/orders"`, `"POST /api/sales/orders/{id}/confirm"`,
        each linked `[:DEPENDS_ON]->(:Module {name: "Sales"})`.

    - **description** — optional summary embedded into Pinecone.
      Example: `"Sales public REST API v1 — order lifecycle endpoints."`

    - **metadata** — freeform. Example: `{"version": "1.4", "owner": "sales-platform"}`
    """
    payload.project = await get_project_service().require_async(payload.project)
    svc = get_ingestion_service()
    doc_id, indexed, nodes, rels = await svc.ingest_api_spec(payload)
    return IngestResponse(
        doc_id=doc_id, chunks_indexed=indexed, nodes_upserted=nodes, relationships_upserted=rels
    )


_TEXT_FORMATS = {"openapi", "postman", "routes", "text"}


@router.post(
    "/apis/normalize",
    response_model=NormalizeApiSpecResponse,
    summary="Convert any API-surface description to a normalized spec JSON",
    description=(
        "Takes user input in one of several shapes and returns an OpenAPI-ish "
        "`{paths: {...}}` JSON ready for POST /ingest/apis. No data is written "
        "to Pinecone or Neo4j — this is a preview/normalization step.\n\n"
        "Supported `format` values:\n"
        "- **openapi** — paste an OpenAPI 3 / Swagger 2 JSON (text or file).\n"
        "- **postman** — paste a Postman Collection v2.1 JSON (text or file).\n"
        "- **routes** — paste a routes/controller file; pass `framework` too. "
        f"Supported frameworks: {', '.join(SUPPORTED_FRAMEWORKS)}.\n"
        "- **text** — plain text list, one endpoint per line. "
        "Accepts `METHOD /path - summary` or `/path METHOD: summary`.\n"
        "- **image** — upload a screenshot of API docs (PNG/JPEG/WEBP). "
        "Uses the vision LLM. Always review before ingesting."
    ),
)
@limiter.limit(settings.rate_limit_ingest)
async def normalize_api_spec(
    request: Request,
    format: str = Form(..., description="One of: openapi, postman, routes, text, image"),
    content: str | None = Form(
        default=None,
        description="Raw text payload for openapi/postman/routes/text. Provide this OR `file`.",
    ),
    file: UploadFile | None = File(
        default=None,
        description="File payload. JSON/text for openapi/postman/routes/text; image for `image`.",
    ),
    framework: str | None = Form(
        default=None,
        description=f"Required when format=routes. One of: {', '.join(SUPPORTED_FRAMEWORKS)}.",
    ),
) -> NormalizeApiSpecResponse:
    fmt = (format or "").lower().strip()
    if fmt not in _TEXT_FORMATS and fmt != "image":
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported format '{format}'. Use: openapi, postman, routes, text, image.",
        )

    try:
        if fmt == "image":
            if file is None:
                raise HTTPException(status_code=422, detail="`file` is required for format=image")
            raw, mime = await read_and_validate(file, kinds={"image"})
            spec = await normalize_image(raw, mime)
        else:
            text_payload = await _read_text_payload(content, file)
            if fmt == "openapi":
                try:
                    parsed = json.loads(text_payload)
                except json.JSONDecodeError as exc:
                    raise HTTPException(status_code=422, detail=f"Invalid JSON: {exc.msg}") from exc
                spec = from_openapi(parsed)
            elif fmt == "postman":
                try:
                    parsed = json.loads(text_payload)
                except json.JSONDecodeError as exc:
                    raise HTTPException(status_code=422, detail=f"Invalid JSON: {exc.msg}") from exc
                spec = from_postman(parsed)
            elif fmt == "routes":
                if not framework:
                    raise HTTPException(
                        status_code=422,
                        detail=(
                            "`framework` is required for format=routes. "
                            f"Supported: {', '.join(SUPPORTED_FRAMEWORKS)}"
                        ),
                    )
                spec = from_routes(text_payload, framework)
            else:  # text
                spec = from_text(text_payload)
    except IngestionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except LLMQuotaError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except LLMModelNotFoundError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    paths = spec.get("paths") or {}
    endpoints_count = sum(
        1
        for methods in paths.values()
        if isinstance(methods, dict)
        for _ in methods
    )
    logger.info(
        f"Normalized API spec: format={fmt} paths={len(paths)} endpoints={endpoints_count}"
    )
    return NormalizeApiSpecResponse(
        spec=spec,
        paths_count=len(paths),
        endpoints_count=endpoints_count,
        format=fmt,
    )


async def _read_text_payload(content: str | None, file: UploadFile | None) -> str:
    if content and content.strip():
        return content
    if file is not None:
        raw = await file.read()
        if not raw:
            raise HTTPException(status_code=422, detail="Uploaded file is empty")
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise HTTPException(
                status_code=422,
                detail="Uploaded file is not valid UTF-8 text",
            ) from exc
    raise HTTPException(status_code=422, detail="Either `content` or `file` is required")
