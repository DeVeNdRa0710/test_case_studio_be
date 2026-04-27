"""
Normalize heterogeneous API-surface descriptions into the minimal OpenAPI-ish
shape this app's ingestion pipeline expects:

    { "paths": { "<path>": { "<method>": { "summary": str, "operationId": str } } } }

Supported inputs:
    - OpenAPI 3 / Swagger 2 JSON
    - Postman Collection v2.1 JSON
    - Routes file (Express/NestJS/Rails/Spring/Flask/FastAPI/Django/Laravel/Go-Gin)
    - Plain text list of endpoints
    - Screenshot of API docs (vision LLM)
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable

from app.core.exceptions import IngestionError
from app.core.logging import logger
from app.services.vision import get_vision

_ALLOWED_METHODS = {"get", "post", "put", "patch", "delete", "head", "options"}

SUPPORTED_FRAMEWORKS = (
    "express",
    "fastify",
    "nestjs",
    "rails",
    "spring",
    "flask",
    "fastapi",
    "django",
    "laravel",
    "go-gin",
)


# ============================================================
# Public API
# ============================================================

def from_openapi(spec: Any) -> dict[str, Any]:
    if not isinstance(spec, dict):
        raise IngestionError("OpenAPI spec must be a JSON object")
    paths = spec.get("paths")
    if not isinstance(paths, dict) or not paths:
        raise IngestionError("Spec has no `paths` object or it is empty")
    out: dict[str, dict[str, Any]] = {}
    for raw_path, methods in paths.items():
        if not isinstance(methods, dict):
            continue
        path = _normalize_path(str(raw_path))
        bucket = out.setdefault(path, {})
        for method, op in methods.items():
            m = str(method).lower()
            if m not in _ALLOWED_METHODS:
                continue
            op = op or {}
            summary = (op.get("summary") or op.get("description") or "").strip()[:200]
            bucket[m] = {
                "summary": summary,
                "operationId": op.get("operationId") or _default_op_id(m, path),
            }
    if not out:
        raise IngestionError("No endpoints found in `paths`")
    return {"paths": out}


def from_postman(collection: Any) -> dict[str, Any]:
    if not isinstance(collection, dict):
        raise IngestionError("Postman collection must be a JSON object")
    items = collection.get("item")
    if not isinstance(items, list) or not items:
        raise IngestionError("Postman collection has no `item` array")
    variables = _postman_variables(collection)
    endpoints: list[tuple[str, str, str]] = []
    for req in _iter_postman_requests(items):
        try:
            method = str(req["request"].get("method", "")).lower()
            if method not in _ALLOWED_METHODS:
                continue
            raw_url = _postman_url_raw(req["request"].get("url"))
            if not raw_url:
                continue
            path = _path_from_url(raw_url, variables)
            if not path:
                continue
            summary = (req.get("name") or "").strip()
            endpoints.append((method.upper(), path, summary))
        except Exception as exc:
            logger.debug(f"skipped postman item: {exc}")
    if not endpoints:
        raise IngestionError("No valid requests found in Postman collection")
    return _endpoints_to_spec(endpoints)


def from_routes(text: str, framework: str) -> dict[str, Any]:
    if not text.strip():
        raise IngestionError("Routes text is empty")
    fw = (framework or "").lower().strip()
    extractors: dict[str, Callable[[str], list[tuple[str, str, str]]]] = {
        "express": _routes_express,
        "fastify": _routes_express,
        "nestjs": _routes_nestjs,
        "rails": _routes_rails,
        "spring": _routes_spring,
        "flask": _routes_flask,
        "fastapi": _routes_fastapi,
        "django": _routes_django,
        "laravel": _routes_laravel,
        "go-gin": _routes_go_gin,
    }
    fn = extractors.get(fw)
    if not fn:
        raise IngestionError(
            f"Unsupported framework '{framework}'. Supported: {', '.join(SUPPORTED_FRAMEWORKS)}"
        )
    endpoints = fn(text)
    if not endpoints:
        raise IngestionError(
            f"No routes matched for framework '{framework}'. "
            "Check that the pasted content is the routes/controller file."
        )
    return _endpoints_to_spec(endpoints)


def from_text(text: str) -> dict[str, Any]:
    if not text.strip():
        raise IngestionError("Text is empty")
    endpoints: list[tuple[str, str, str]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("//"):
            continue
        parsed = _parse_plain_line(line)
        if parsed:
            endpoints.append(parsed)
    if not endpoints:
        raise IngestionError(
            "No endpoints detected. Use lines like 'POST /login — user login'."
        )
    return _endpoints_to_spec(endpoints)


async def from_image(image_bytes: bytes, mime_type: str) -> dict[str, Any]:
    system = (
        "You are an API documentation extractor. The user provides a screenshot "
        "of API documentation (Postman, Swagger UI, internal wiki, a routes table, "
        "etc.). Identify every endpoint visible in the image and return ONE JSON "
        "object in the OpenAPI `paths` shape.\n\n"
        "Strict rules:\n"
        "- Top-level key is exactly `paths` (nothing else at the top level).\n"
        "- Keys under `paths` are endpoint paths (must start with `/`).\n"
        "- Values are objects keyed by lowercase HTTP method (get/post/put/patch/delete).\n"
        "- Each method value MUST have `summary` (<=100 chars) and `operationId` (camelCase).\n"
        "- Do NOT invent endpoints that are not clearly visible.\n"
        "- Do NOT include parameters/requestBody/responses — only summary + operationId.\n"
        "- If nothing is readable, return {\"paths\": {}}."
    )
    user = (
        "Extract every API endpoint visible in this screenshot. Return JSON only."
    )
    vision = get_vision()
    raw = await vision.complete_with_image(
        system=system,
        user=user,
        image_bytes=image_bytes,
        mime_type=mime_type,
        json_mode=True,
    )
    try:
        data = json.loads(_strip_code_fence(raw))
    except Exception as exc:
        raise IngestionError(f"Vision LLM returned non-JSON: {raw[:200]}") from exc
    try:
        return from_openapi(data)
    except IngestionError as exc:
        raise IngestionError(f"Vision LLM output was not a usable spec: {exc}") from exc


# ============================================================
# Helpers
# ============================================================

def _endpoints_to_spec(endpoints: list[tuple[str, str, str]]) -> dict[str, Any]:
    out: dict[str, dict[str, Any]] = {}
    for method, path, summary in endpoints:
        m = method.lower()
        if m not in _ALLOWED_METHODS:
            continue
        bucket = out.setdefault(path, {})
        bucket[m] = {
            "summary": summary.strip()[:200],
            "operationId": _default_op_id(m, path),
        }
    if not out:
        raise IngestionError("No valid endpoints extracted")
    return {"paths": out}


def _default_op_id(method: str, path: str) -> str:
    tokens = re.findall(r"[A-Za-z0-9]+", path)
    if not tokens:
        return method
    camel = tokens[0].lower() + "".join(t.capitalize() for t in tokens[1:])
    return f"{method}{camel[:1].upper()}{camel[1:]}"


def _normalize_path(path: str) -> str:
    path = path.strip()
    if not path.startswith("/"):
        path = "/" + path
    # Django/Flask <int:id> or <id> → {id} (must run BEFORE :param)
    path = re.sub(r"<[^>]*?([A-Za-z_][A-Za-z0-9_]*)>", r"{\1}", path)
    # Express/Rails/Gin :id → {id}
    path = re.sub(r":([A-Za-z_][A-Za-z0-9_]*)", r"{\1}", path)
    return path


def _strip_code_fence(raw: str) -> str:
    s = raw.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9]*\n?", "", s)
        s = re.sub(r"\n?```\s*$", "", s)
    return s


# ---------- plain text ----------

_PLAIN_METHOD_FIRST = re.compile(
    r"^\s*(GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)\s+(\S+)(?:\s*[-—:>]{1,2}\s*(.*))?$",
    re.IGNORECASE,
)
_PLAIN_PATH_FIRST = re.compile(
    r"^\s*(/\S*)\s+(GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)(?:\s*[-—:>]{1,2}\s*(.*))?$",
    re.IGNORECASE,
)


def _parse_plain_line(line: str) -> tuple[str, str, str] | None:
    m = _PLAIN_METHOD_FIRST.match(line)
    if m:
        return m.group(1), _normalize_path(m.group(2)), (m.group(3) or "").strip()
    m = _PLAIN_PATH_FIRST.match(line)
    if m:
        return m.group(2), _normalize_path(m.group(1)), (m.group(3) or "").strip()
    return None


# ---------- postman ----------

def _postman_variables(collection: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for v in collection.get("variable", []) or []:
        key = v.get("key")
        val = v.get("value")
        if isinstance(key, str) and isinstance(val, str):
            out[key] = val
    return out


def _iter_postman_requests(items: list[Any]):
    for item in items:
        if not isinstance(item, dict):
            continue
        if isinstance(item.get("item"), list):
            yield from _iter_postman_requests(item["item"])
        elif isinstance(item.get("request"), dict):
            yield item


def _postman_url_raw(url: Any) -> str:
    if isinstance(url, str):
        return url
    if isinstance(url, dict):
        raw = url.get("raw")
        if isinstance(raw, str) and raw.strip():
            return raw
        host = url.get("host")
        path = url.get("path")
        host_s = "/".join(host) if isinstance(host, list) else (host or "")
        path_s = "/".join(path) if isinstance(path, list) else (path or "")
        if path_s:
            return f"{host_s}/{path_s}" if host_s else f"/{path_s}"
    return ""


def _path_from_url(raw: str, variables: dict[str, str]) -> str:
    url = raw.strip()
    url = re.sub(
        r"\{\{([^{}]+)\}\}",
        lambda m: variables.get(m.group(1).strip(), ""),
        url,
    )
    m = re.match(r"^(?:https?://)?[^/]+(/.*)$", url)
    if m:
        url = m.group(1)
    if not url.startswith("/"):
        url = "/" + url
    url = url.split("?", 1)[0].split("#", 1)[0]
    if len(url) > 1 and url.endswith("/"):
        url = url[:-1]
    return _normalize_path(url) if url else ""


# ---------- framework route extractors ----------

def _routes_express(text: str) -> list[tuple[str, str, str]]:
    pattern = re.compile(
        r"\b(?:app|router|api|routes?)\s*\.\s*"
        r"(get|post|put|patch|delete|head|options)"
        r"\s*\(\s*(?P<q>['\"`])(?P<path>[^'\"`]+)(?P=q)",
        re.IGNORECASE,
    )
    return [
        (m.group(1), _normalize_path(m.group("path")), "")
        for m in pattern.finditer(text)
    ]


def _routes_nestjs(text: str) -> list[tuple[str, str, str]]:
    pattern = re.compile(
        r"@(Get|Post|Put|Patch|Delete|Head|Options)\s*\(\s*"
        r"(?:(?P<q>['\"])(?P<path>[^'\"]*)(?P=q))?",
        re.IGNORECASE,
    )
    out: list[tuple[str, str, str]] = []
    for m in pattern.finditer(text):
        path = m.group("path") or "/"
        out.append((m.group(1), _normalize_path(path), ""))
    return out


def _routes_rails(text: str) -> list[tuple[str, str, str]]:
    out: list[tuple[str, str, str]] = []
    explicit = re.compile(
        r"^\s*(get|post|put|patch|delete|head|options)\s+['\"]([^'\"]+)['\"]",
        re.IGNORECASE | re.MULTILINE,
    )
    for m in explicit.finditer(text):
        out.append((m.group(1), _normalize_path(m.group(2)), ""))

    resources = re.compile(
        r"^\s*resources\s+:([a-z_][a-z0-9_]*)",
        re.IGNORECASE | re.MULTILINE,
    )
    for m in resources.finditer(text):
        name = m.group(1)
        base, member = f"/{name}", f"/{name}/{{id}}"
        out.extend([
            ("GET", base, f"List {name}"),
            ("POST", base, f"Create {name}"),
            ("GET", member, f"Show {name}"),
            ("PUT", member, f"Update {name}"),
            ("PATCH", member, f"Update {name}"),
            ("DELETE", member, f"Destroy {name}"),
        ])

    singleton = re.compile(
        r"^\s*resource\s+:([a-z_][a-z0-9_]*)",
        re.IGNORECASE | re.MULTILINE,
    )
    for m in singleton.finditer(text):
        name = m.group(1)
        base = f"/{name}"
        out.extend([
            ("GET", base, f"Show {name}"),
            ("POST", base, f"Create {name}"),
            ("PUT", base, f"Update {name}"),
            ("PATCH", base, f"Update {name}"),
            ("DELETE", base, f"Destroy {name}"),
        ])
    return out


def _routes_spring(text: str) -> list[tuple[str, str, str]]:
    out: list[tuple[str, str, str]] = []
    simple = re.compile(
        r"@(Get|Post|Put|Patch|Delete)Mapping\s*\(\s*"
        r"(?:value\s*=\s*)?(?:\{\s*)?"
        r"(?P<q>['\"])(?P<path>[^'\"]*)(?P=q)",
        re.IGNORECASE,
    )
    for m in simple.finditer(text):
        out.append((m.group(1), _normalize_path(m.group("path") or "/"), ""))

    full = re.compile(
        r"@RequestMapping\s*\([^)]*?"
        r"(?:value\s*=\s*)?(?P<q>['\"])(?P<path>[^'\"]*)(?P=q)"
        r"[^)]*?method\s*=\s*RequestMethod\.(GET|POST|PUT|PATCH|DELETE)",
        re.IGNORECASE | re.DOTALL,
    )
    for m in full.finditer(text):
        out.append((m.group(3), _normalize_path(m.group("path") or "/"), ""))
    return out


def _routes_flask(text: str) -> list[tuple[str, str, str]]:
    out: list[tuple[str, str, str]] = []
    pattern = re.compile(
        r"@(?:app|bp|blueprint|api)\.route\s*\(\s*"
        r"(?P<q>['\"])(?P<path>[^'\"]+)(?P=q)(?P<rest>[^)]*)\)",
        re.IGNORECASE | re.DOTALL,
    )
    for m in pattern.finditer(text):
        path = m.group("path")
        rest = m.group("rest") or ""
        methods_m = re.search(r"methods\s*=\s*\[([^\]]+)\]", rest)
        if methods_m:
            for tok in re.findall(r"['\"]([A-Za-z]+)['\"]", methods_m.group(1)):
                out.append((tok.upper(), _normalize_path(path), ""))
        else:
            out.append(("GET", _normalize_path(path), ""))

    shortcut = re.compile(
        r"@(?:app|bp|blueprint|api)\."
        r"(get|post|put|patch|delete|head|options)"
        r"\s*\(\s*(?P<q>['\"])(?P<path>[^'\"]+)(?P=q)",
        re.IGNORECASE,
    )
    for m in shortcut.finditer(text):
        out.append((m.group(1), _normalize_path(m.group("path")), ""))
    return out


def _routes_fastapi(text: str) -> list[tuple[str, str, str]]:
    pattern = re.compile(
        r"@(?:app|router|api)\."
        r"(get|post|put|patch|delete|head|options)"
        r"\s*\(\s*(?P<q>['\"])(?P<path>[^'\"]+)(?P=q)",
        re.IGNORECASE,
    )
    return [
        (m.group(1), _normalize_path(m.group("path")), "")
        for m in pattern.finditer(text)
    ]


def _routes_django(text: str) -> list[tuple[str, str, str]]:
    pattern = re.compile(
        r"(?:path|re_path)\s*\(\s*(?P<q>['\"])(?P<path>[^'\"]+)(?P=q)",
        re.IGNORECASE,
    )
    return [
        ("GET", _normalize_path(m.group("path")), "")
        for m in pattern.finditer(text)
    ]


def _routes_laravel(text: str) -> list[tuple[str, str, str]]:
    pattern = re.compile(
        r"Route::(get|post|put|patch|delete|head|options)\s*\(\s*"
        r"(?P<q>['\"])(?P<path>[^'\"]+)(?P=q)",
        re.IGNORECASE,
    )
    return [
        (m.group(1), _normalize_path(m.group("path")), "")
        for m in pattern.finditer(text)
    ]


def _routes_go_gin(text: str) -> list[tuple[str, str, str]]:
    pattern = re.compile(
        r"\b(?:r|router|e|g|api|mux)\s*\.\s*"
        r"(GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)\s*\(\s*"
        r"(?P<q>['\"])(?P<path>[^'\"]+)(?P=q)",
        re.IGNORECASE,
    )
    return [
        (m.group(1), _normalize_path(m.group("path")), "")
        for m in pattern.finditer(text)
    ]
