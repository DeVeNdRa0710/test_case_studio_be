import asyncio
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
    FigmaBatchResponse,
    FigmaBatchResult,
    FigmaIngestRequest,
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
from app.services.figma_extractor import get_figma_screenshot_extractor
from app.services.ingestion_service import get_ingestion_service
from app.services.project_service import get_project_service
from app.services.vision import LLMModelNotFoundError, LLMQuotaError
from app.utils.pdf import extract_text_from_pdf
from app.utils.uploads import assert_pdf_page_limit, read_and_validate

router = APIRouter(prefix="/ingest", tags=["ingest"])


class FigmaFromImageResponse(BaseModel):
    ok: bool = True
    figma_json: dict[str, Any]
    screen_name: str
    module: str


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

    if not project or not module or not title:
        raise HTTPException(
            status_code=422, detail="`project`, `module`, and `title` are required"
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
    module: str = Form(..., description="ERP module applied to every file."),
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
    if not project.strip() or not module.strip():
        raise HTTPException(status_code=422, detail="`project` and `module` are required")
    project = await get_project_service().require_async(project)

    svc = get_ingestion_service()
    results: list[BatchDocResult] = []
    totals = BatchTotals()
    prefix = (title_prefix or "").strip()

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
                module=module.strip(),
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
    "/figma",
    response_model=IngestResponse,
    summary="Ingest a Figma screen (preprocessed JSON)",
    description=(
        "Takes a preprocessed Figma export and registers the screen in both stores.\n\n"
        "- Pinecone: the flattened UI element tree is chunked + embedded.\n"
        "- Neo4j: creates a (:UIScreen {name: screen_name})-[:DEPENDS_ON]->(:Module) edge "
        "and extracts any additional Entity/API references mentioned in the description."
    ),
)
@limiter.limit(settings.rate_limit_ingest)
async def ingest_figma(request: Request, payload: FigmaIngestRequest) -> IngestResponse:  # noqa: D401
    """
    Field-by-field example:

    - **module** — ERP module this screen belongs to.
      Example: `"Sales"`

    - **screen_name** — canonical name. Becomes a unique :UIScreen node.
      Example: `"NewOrderScreen"`, `"CustomerList"`, `"InvoiceDetail"`

    - **figma_json** — the preprocessed Figma tree. The service walks
      `name` / `type` / `children`.
      Example:
      ```json
      {
        "name": "NewOrder",
        "type": "FRAME",
        "children": [
          {"type": "INPUT",  "name": "customer"},
          {"type": "INPUT",  "name": "itemSku"},
          {"type": "BUTTON", "name": "confirm"},
          {"type": "LABEL",  "name": "orderTotal"}
        ]
      }
      ```

    - **metadata** — freeform. Example: `{"figma_file": "abc123", "version": "v3"}`

    Full payload example:
    ```json
    {
      "module": "Sales",
      "screen_name": "NewOrderScreen",
      "figma_json": { "...": "..." },
      "metadata": {"figma_node_id": "1:42"}
    }
    ```
    """
    payload.project = await get_project_service().require_async(payload.project)
    svc = get_ingestion_service()
    doc_id, indexed, nodes, rels = await svc.ingest_figma(payload)
    return IngestResponse(
        doc_id=doc_id, chunks_indexed=indexed, nodes_upserted=nodes, relationships_upserted=rels
    )


@router.post(
    "/figma/from-image",
    response_model=FigmaFromImageResponse,
    summary="Auto-generate Figma JSON from a screenshot (vision LLM)",
    description=(
        "Upload a screenshot of a UI (web page, mobile screen, or Figma design). "
        "A vision LLM identifies the visible elements and returns a simplified "
        "Figma-style JSON tree {name, type, children}. The response is NOT ingested "
        "automatically — the user reviews/edits it, then calls POST /ingest/figma."
    ),
)
@limiter.limit(settings.rate_limit_vision)
async def figma_from_image(
    request: Request,
    file: UploadFile = File(
        ...,
        description="Screenshot of the UI. Accepts PNG, JPEG, or WEBP.",
    ),
    screen_name: str = Form(
        default="Screen",
        description="Canonical name for the extracted screen (e.g. 'SearchPage').",
    ),
    module: str = Form(
        default="",
        description="Module this screen belongs to (e.g. 'Catalog'). Optional here; required on final ingest.",
    ),
) -> FigmaFromImageResponse:
    raw, mime = await read_and_validate(file, kinds={"image"})

    extractor = get_figma_screenshot_extractor()
    try:
        tree = await extractor.extract(
            image_bytes=raw,
            mime_type=mime,
            screen_name=screen_name.strip() or "Screen",
            module=module.strip(),
        )
    except LLMQuotaError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except LLMModelNotFoundError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    logger.info(
        f"Figma JSON extracted from image: screen={screen_name} "
        f"root_children={len(tree.get('children', []))}"
    )
    return FigmaFromImageResponse(
        figma_json=tree,
        screen_name=screen_name.strip() or "Screen",
        module=module.strip(),
    )


_FIGMA_BATCH_ALLOWED_MIMES = {"image/png", "image/jpeg", "image/jpg", "image/webp"}
_FIGMA_BATCH_MAX_BYTES = 8 * 1024 * 1024
_FIGMA_BATCH_MAX_FILES = 20
_FIGMA_BATCH_CONCURRENCY = 4


def _parse_repeatable_form(raw: str | None, count: int) -> list[str]:
    """Accept either a JSON array or a newline/comma-separated list; pad to `count`."""
    if not raw:
        return [""] * count
    s = raw.strip()
    values: list[str] = []
    if s.startswith("["):
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                values = [str(v or "").strip() for v in parsed]
        except json.JSONDecodeError:
            values = []
    if not values:
        values = [part.strip() for part in s.replace("\r", "").split("\n") if part.strip()]
        if len(values) <= 1:
            values = [part.strip() for part in s.split(",") if part.strip()]
    if len(values) < count:
        values = values + [""] * (count - len(values))
    return values[:count]


def _default_screen_name(file_name: str) -> str:
    base = file_name.rsplit(".", 1)[0] if "." in file_name else file_name
    return base.strip() or "Screen"


@router.post(
    "/figma/from-images",
    response_model=FigmaBatchResponse,
    summary="Extract Figma JSON from multiple screenshots (optionally auto-ingest)",
    description=(
        "Upload 1..N UI screenshots. Each image is sent to the vision LLM to produce "
        "a Figma-style JSON tree, and (when `auto_ingest=true`, the default) each tree "
        "is written to Pinecone + Neo4j as a :UIScreen.\n\n"
        "- Vision calls run concurrently (capped) to stay under model rate limits.\n"
        "- A failure on one image does not abort the batch — failed files appear in "
        "  `results` with `ok=false` and an `error` message.\n"
        "- `screen_names` and `modules` are optional; if omitted, the filename "
        "  (without extension) is used as the screen name and `default_module` as the module.\n"
        "- `screen_names` and `modules` accept either a JSON array string "
        "  (`[\"Login\",\"Home\"]`) or newline/comma-separated values."
    ),
)
@limiter.limit(settings.rate_limit_vision)
async def ingest_figma_from_images(
    request: Request,
    files: list[UploadFile] = File(
        ...,
        description="One or more UI screenshots. Use the same field name 'files' for each.",
    ),
    project: str = Form(..., description="Project (tenant) to ingest into."),
    default_module: str = Form(
        "",
        description="Module used when `modules` is not provided per-file. Required if `auto_ingest=true` and no per-file modules are given.",
    ),
    screen_names: str | None = Form(
        default=None,
        description="Optional per-file screen names (JSON array or newline/comma-separated). Missing entries fall back to the filename.",
    ),
    modules: str | None = Form(
        default=None,
        description="Optional per-file modules (JSON array or newline/comma-separated). Missing entries fall back to `default_module`.",
    ),
    auto_ingest: bool = Form(
        default=True,
        description="When true (default), each extracted tree is written to Pinecone + Neo4j immediately.",
    ),
) -> FigmaBatchResponse:
    if not files:
        raise HTTPException(status_code=422, detail="At least one file is required")
    if len(files) > _FIGMA_BATCH_MAX_FILES:
        raise HTTPException(
            status_code=413,
            detail=f"Too many files: {len(files)} (max {_FIGMA_BATCH_MAX_FILES} per request)",
        )
    if not project.strip():
        raise HTTPException(status_code=422, detail="`project` is required")
    project = await get_project_service().require_async(project)

    per_file_names = _parse_repeatable_form(screen_names, len(files))
    per_file_modules = _parse_repeatable_form(modules, len(files))
    default_module_clean = (default_module or "").strip()

    # Preload file bytes once (UploadFile reads must happen on the request task).
    loaded: list[tuple[str, str, bytes, str | None]] = []  # (file_name, mime, bytes, error)
    for f in files:
        file_name = f.filename or "upload"
        try:
            raw, mime = await read_and_validate(f, kinds={"image"})
            loaded.append((file_name, mime, raw, None))
        except HTTPException as exc:
            loaded.append((file_name, (f.content_type or "").lower(), b"", str(exc.detail)))

    extractor = get_figma_screenshot_extractor()
    svc = get_ingestion_service()
    sem = asyncio.Semaphore(_FIGMA_BATCH_CONCURRENCY)

    async def process_one(idx: int) -> FigmaBatchResult:
        file_name, mime, raw, load_error = loaded[idx]
        screen_name = (per_file_names[idx] or "").strip() or _default_screen_name(file_name)
        module = (per_file_modules[idx] or "").strip() or default_module_clean

        if load_error:
            return FigmaBatchResult(
                ok=False,
                file_name=file_name,
                screen_name=screen_name,
                module=module,
                error=load_error,
            )

        async with sem:
            try:
                tree = await extractor.extract(
                    image_bytes=raw,
                    mime_type=mime,
                    screen_name=screen_name,
                    module=module,
                )
            except LLMQuotaError as exc:
                return FigmaBatchResult(
                    ok=False,
                    file_name=file_name,
                    screen_name=screen_name,
                    module=module,
                    error=f"Vision LLM quota exhausted: {exc}",
                )
            except LLMModelNotFoundError as exc:
                return FigmaBatchResult(
                    ok=False,
                    file_name=file_name,
                    screen_name=screen_name,
                    module=module,
                    error=f"Vision LLM not configured: {exc}",
                )
            except Exception as exc:  # noqa: BLE001 - we want per-file isolation
                logger.error(f"Figma vision extract failed for {file_name}: {exc}")
                return FigmaBatchResult(
                    ok=False,
                    file_name=file_name,
                    screen_name=screen_name,
                    module=module,
                    error=str(exc),
                )

        if not auto_ingest:
            return FigmaBatchResult(
                ok=True,
                file_name=file_name,
                screen_name=screen_name,
                module=module,
                figma_json=tree,
                ingested=False,
            )

        if not module:
            return FigmaBatchResult(
                ok=False,
                file_name=file_name,
                screen_name=screen_name,
                module=module,
                figma_json=tree,
                error="auto_ingest=true requires a module — pass `default_module` or per-file `modules`.",
            )

        try:
            doc_id, indexed, nodes, rels = await svc.ingest_figma(
                FigmaIngestRequest(
                    project=project,
                    module=module,
                    screen_name=screen_name,
                    figma_json=tree,
                    metadata={"source": "batch_from_image", "file_name": file_name},
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Figma ingest failed for {file_name}: {exc}")
            return FigmaBatchResult(
                ok=False,
                file_name=file_name,
                screen_name=screen_name,
                module=module,
                figma_json=tree,
                error=f"ingest failed: {exc}",
            )

        return FigmaBatchResult(
            ok=True,
            file_name=file_name,
            screen_name=screen_name,
            module=module,
            figma_json=tree,
            ingested=True,
            doc_id=doc_id,
            chunks_indexed=indexed,
            nodes_upserted=nodes,
            relationships_upserted=rels,
        )

    results = await asyncio.gather(*(process_one(i) for i in range(len(files))))

    totals = BatchTotals()
    for r in results:
        totals.chunks_indexed += r.chunks_indexed
        totals.nodes_upserted += r.nodes_upserted
        totals.relationships_upserted += r.relationships_upserted
    succeeded = sum(1 for r in results if r.ok)

    logger.info(
        f"Figma batch: project={project} files={len(files)} "
        f"succeeded={succeeded} failed={len(files) - succeeded} "
        f"auto_ingest={auto_ingest}"
    )

    return FigmaBatchResponse(
        ok=succeeded > 0,
        total_files=len(files),
        succeeded=succeeded,
        failed=len(files) - succeeded,
        results=list(results),
        totals=totals,
    )


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
