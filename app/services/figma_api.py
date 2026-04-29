"""
Fetch design data directly from Figma's REST API.

Flow: caller provides a `figma.com/design/<fileKey>/...?node-id=<nodeId>` URL +
a personal access token. We parse the URL, call Figma's `/v1/files/{fileKey}`
(or `/v1/files/{fileKey}/nodes?ids=...` when a node is targeted), then collapse
the rich Figma node tree into the simplified `{name, type, children}` shape
that `IngestionService.ingest_figma` already understands.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qs, urlparse

import httpx

from app.core.config import settings
from app.core.exceptions import AppError, IngestionError
from app.core.logging import logger
from app.core.resilience import llm_breaker, retryable_external, with_breaker


# Figma element types we want to surface as input-like / interactive primitives
# in the simplified tree. Anything else falls through with its raw type.
_INTERACTIVE_TYPE_HINTS: dict[str, str] = {
    "input": "INPUT",
    "textfield": "INPUT",
    "text field": "INPUT",
    "textbox": "INPUT",
    "search": "INPUT",
    "search bar": "INPUT",
    "field": "INPUT",
    "button": "BUTTON",
    "btn": "BUTTON",
    "cta": "BUTTON",
    "submit": "BUTTON",
    "checkbox": "CHECKBOX",
    "radio": "RADIO",
    "toggle": "TOGGLE",
    "switch": "TOGGLE",
    "dropdown": "SELECT",
    "select": "SELECT",
    "combobox": "SELECT",
    "link": "LINK",
    "tab": "TAB",
    "tabs": "TAB",
}

_DROP_TYPES = {"VECTOR", "BOOLEAN_OPERATION", "STAR", "LINE", "ELLIPSE", "REGULAR_POLYGON"}


class FigmaAuthError(AppError):
    status_code = 401
    code = "figma_auth_error"


class FigmaNotFoundError(AppError):
    status_code = 404
    code = "figma_not_found"


@dataclass(frozen=True)
class FigmaUrlParts:
    file_key: str
    node_id: str | None  # already in Figma's "1:23" form, not "1-23"


_FILE_KEY_RE = re.compile(
    r"figma\.com/(?:design|file|board|proto)/(?P<key>[a-zA-Z0-9]+)"
)
_BRANCH_RE = re.compile(r"/branch/(?P<branch>[a-zA-Z0-9]+)/")


def parse_figma_url(url: str) -> FigmaUrlParts:
    """
    Pull fileKey + nodeId from any common Figma URL shape:
      https://www.figma.com/design/<fileKey>/<name>?node-id=1-23
      https://www.figma.com/file/<fileKey>/<name>
      https://www.figma.com/design/<fileKey>/branch/<branchKey>/<name>?node-id=...
    Branch URLs use the branchKey when fetching.
    """
    if not url or not url.strip():
        raise IngestionError("figma_url is required")

    parsed = urlparse(url.strip())
    if not parsed.netloc.endswith("figma.com"):
        raise IngestionError(f"Not a Figma URL: {url!r}")

    m = _FILE_KEY_RE.search(parsed.path) or _FILE_KEY_RE.search(url)
    if not m:
        raise IngestionError(
            "Could not find fileKey in URL. "
            "Expected /design/<fileKey>/... or /file/<fileKey>/..."
        )
    file_key = m.group("key")

    branch_match = _BRANCH_RE.search(parsed.path)
    if branch_match:
        file_key = branch_match.group("branch")

    node_id: str | None = None
    qs = parse_qs(parsed.query)
    raw_node = (qs.get("node-id") or qs.get("nodeId") or [None])[0]
    if raw_node:
        node_id = raw_node.replace("-", ":")

    return FigmaUrlParts(file_key=file_key, node_id=node_id)


class FigmaApiClient:
    def __init__(self, token: str | None = None) -> None:
        self._token = (token or "").strip() or None
        self._base = settings.figma_api_base.rstrip("/")
        self._timeout = settings.figma_http_timeout

    @retryable_external()
    @with_breaker(llm_breaker, "Figma")
    async def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        url = f"{self._base}{path}"
        headers = {"X-Figma-Token": self._token} if self._token else {}
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(url, headers=headers, params=params)
        if resp.status_code == 401 or resp.status_code == 403:
            if not self._token:
                raise FigmaAuthError(
                    "Figma's API requires a Personal Access Token. "
                    "Get one at figma.com → Settings → Security → Personal access tokens → "
                    "'Generate new token' (scope: File content read). "
                    "Then paste it in the 'Figma access token' field, or set FIGMA_TOKEN on the server."
                )
            raise FigmaAuthError(
                f"Figma rejected the token ({resp.status_code}). The token may be expired, "
                "revoked, or lack 'File content: read' scope, or your account may not have access "
                "to this file. Generate a new token at figma.com → Settings → Security → "
                "Personal access tokens."
            )
        if resp.status_code == 404:
            raise FigmaNotFoundError(
                "Figma file or node not found. Verify the URL and that the "
                "token has access to the target file."
            )
        if resp.status_code >= 400:
            raise IngestionError(
                f"Figma API error {resp.status_code}: {resp.text[:300]}"
            )
        return resp.json()

    async def fetch_node(self, file_key: str, node_id: str) -> tuple[dict[str, Any], str]:
        """Fetch a specific node (frame/component) from a file. Returns (document, file_name)."""
        data = await self._get(
            f"/v1/files/{file_key}/nodes",
            params={"ids": node_id, "geometry": "paths"},
        )
        file_name = (data.get("name") or "").strip()
        nodes = data.get("nodes") or {}
        # Figma returns the requested id back with ":" preserved.
        entry = nodes.get(node_id) or next(iter(nodes.values()), None)
        if not entry or not entry.get("document"):
            raise FigmaNotFoundError(
                f"Node {node_id} not found in file {file_key}"
            )
        return entry["document"], file_name

    async def fetch_file_root(self, file_key: str) -> tuple[dict[str, Any], str]:
        """Fetch the file's top-level document. Returns (document, file_name)."""
        data = await self._get(f"/v1/files/{file_key}", params={"geometry": "paths"})
        document = data.get("document")
        if not document:
            raise FigmaNotFoundError(f"File {file_key} returned no document")
        file_name = (data.get("name") or "").strip()
        return document, file_name


def _classify_type(raw_type: str, name: str | None) -> str:
    """Map Figma's raw type + element name to the simplified ingestion vocab."""
    rt = (raw_type or "").upper()
    nm = (name or "").lower()

    for hint, mapped in _INTERACTIVE_TYPE_HINTS.items():
        if hint in nm:
            return mapped

    if rt in {"TEXT"}:
        return "LABEL"
    if rt in {"FRAME", "GROUP", "SECTION", "CANVAS"}:
        return "FRAME"
    if rt in {"COMPONENT", "COMPONENT_SET", "INSTANCE"}:
        return "COMPONENT"
    if rt in {"RECTANGLE"}:
        return "RECTANGLE"
    return rt or "NODE"


def _simplify_node(node: dict[str, Any]) -> dict[str, Any] | None:
    """Recursively turn a Figma document node into {name, type, children, ...}."""
    raw_type = node.get("type") or ""
    if raw_type in _DROP_TYPES:
        return None

    name = node.get("name") or ""
    simplified: dict[str, Any] = {
        "name": name,
        "type": _classify_type(raw_type, name),
    }

    if raw_type == "TEXT":
        characters = node.get("characters")
        if characters:
            simplified["text"] = characters[:200]

    children: list[dict[str, Any]] = []
    for child in node.get("children") or []:
        s = _simplify_node(child)
        if s is not None:
            children.append(s)
    if children:
        simplified["children"] = children

    return simplified


def normalize_figma_document(document: dict[str, Any], screen_name: str) -> dict[str, Any]:
    """
    Wrap a Figma document node into the `{name, type, children}` shape used by
    `IngestionService.ingest_figma`. The root name is forced to `screen_name`.
    """
    simplified = _simplify_node(document) or {"name": screen_name, "type": "FRAME"}
    if screen_name:
        simplified["name"] = screen_name
    simplified.setdefault("type", "FRAME")
    simplified.setdefault("children", [])
    return simplified


@dataclass(frozen=True)
class FigmaFetchResult:
    tree: dict[str, Any]
    parts: FigmaUrlParts
    file_name: str
    node_name: str | None
    derived_screen_name: str
    derived_module: str


def _slugify_module(name: str) -> str:
    cleaned = "".join(c if c.isalnum() or c in {" ", "-", "_"} else " " for c in name)
    cleaned = " ".join(cleaned.split())
    return cleaned or "Figma"


async def figma_url_to_tree(
    *,
    figma_url: str,
    token: str | None = None,
    screen_name: str | None = None,
    module: str | None = None,
) -> FigmaFetchResult:
    """Parse URL → fetch Figma → normalize. Derives screen_name/module from Figma metadata when omitted."""
    parts = parse_figma_url(figma_url)
    client = FigmaApiClient(token=token)
    if parts.node_id:
        document, file_name = await client.fetch_node(parts.file_key, parts.node_id)
    else:
        document, file_name = await client.fetch_file_root(parts.file_key)

    node_name = (document.get("name") or "").strip() or None
    derived_screen = (
        (screen_name or "").strip()
        or (node_name if parts.node_id else None)
        or file_name
        or "Screen"
    )
    derived_module = (module or "").strip() or _slugify_module(file_name or "Figma")

    tree = normalize_figma_document(document, screen_name=derived_screen)
    logger.info(
        f"Figma fetched: file={parts.file_key} node={parts.node_id or '<root>'} "
        f"file_name={file_name!r} screen={derived_screen!r} module={derived_module!r} "
        f"root_children={len(tree.get('children', []))}"
    )
    return FigmaFetchResult(
        tree=tree,
        parts=parts,
        file_name=file_name,
        node_name=node_name,
        derived_screen_name=derived_screen,
        derived_module=derived_module,
    )
