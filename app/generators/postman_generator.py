import json
import re
import uuid
from typing import Any

from app.schemas.testcase import TestCase


def _as_url_parts(path: str, base_url: str | None) -> dict[str, Any]:
    base = (base_url or "").rstrip("/")
    full = f"{base}{path}" if path.startswith("/") else f"{base}/{path}" if base else path
    raw = full or path
    host: list[str] = []
    path_parts: list[str] = []
    protocol = "https"

    m = re.match(r"^(https?)://([^/]+)(/.*)?$", raw)
    if m:
        protocol = m.group(1)
        host = m.group(2).split(".")
        path_parts = [p for p in (m.group(3) or "").split("/") if p]
    else:
        path_parts = [p for p in raw.split("/") if p]

    return {
        "raw": raw or path,
        "protocol": protocol,
        "host": host or ["{{baseUrl}}"],
        "path": path_parts,
    }


def _build_item(api: dict[str, Any], base_url: str | None) -> dict[str, Any]:
    method = (api.get("method") or "GET").upper()
    name = api.get("name") or f"{method} {api.get('path', '')}"
    path = api.get("path") or "/"
    body = api.get("body")
    expected_status = int(api.get("expected_status") or 200)
    save_as = api.get("save_as")

    request_obj: dict[str, Any] = {
        "method": method,
        "header": [{"key": "Content-Type", "value": "application/json"}],
        "url": _as_url_parts(path, base_url),
    }
    if body is not None:
        request_obj["body"] = {
            "mode": "raw",
            "raw": json.dumps(body, indent=2),
            "options": {"raw": {"language": "json"}},
        }

    test_lines = [
        f"pm.test(\"Status {expected_status}\", function () {{",
        f"    pm.response.to.have.status({expected_status});",
        "});",
    ]
    if save_as:
        test_lines += [
            "try {",
            "    var __json = pm.response.json();",
            f"    if (__json && __json.id) pm.collectionVariables.set(\"{save_as}\", __json.id);",
            f"    else pm.collectionVariables.set(\"{save_as}\", JSON.stringify(__json));",
            "} catch (e) {}",
        ]

    return {
        "name": name,
        "request": request_obj,
        "response": [],
        "event": [
            {
                "listen": "test",
                "script": {"type": "text/javascript", "exec": test_lines},
            }
        ],
    }


class PostmanGenerator:
    def generate(
        self,
        test_cases: list[TestCase],
        *,
        collection_name: str = "ERP Generated Collection",
        base_url: str | None = None,
    ) -> dict[str, Any]:
        items: list[dict[str, Any]] = []
        for tc in test_cases:
            folder = {
                "name": tc.scenario or "Scenario",
                "item": [_build_item(api, base_url) for api in tc.apis],
            }
            if folder["item"]:
                items.append(folder)

        return {
            "info": {
                "_postman_id": str(uuid.uuid4()),
                "name": collection_name,
                "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json",
            },
            "item": items,
            "variable": [{"key": "baseUrl", "value": base_url or ""}],
        }


_pm_gen: PostmanGenerator | None = None


def get_postman_generator() -> PostmanGenerator:
    global _pm_gen
    if _pm_gen is None:
        _pm_gen = PostmanGenerator()
    return _pm_gen
