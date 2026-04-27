"""
Upload validation: size caps, magic-byte sniffing, and PDF page-count caps.
Uses `filetype` (pure-Python, zero system deps) instead of python-magic.
"""

from __future__ import annotations

from io import BytesIO
from typing import Iterable

import filetype
from fastapi import HTTPException, UploadFile

from app.core.config import settings


_TEXT_FALLBACK_EXTS = {".txt", ".md", ".json", ".yaml", ".yml", ".csv"}

_KIND_MIMES: dict[str, set[str]] = {
    "pdf": {"application/pdf"},
    "image": {"image/png", "image/jpeg", "image/jpg", "image/webp"},
    "text": {"text/plain", "text/markdown", "application/json", "text/csv"},
    "json": {"application/json"},
}


def _looks_like_text(raw: bytes, sample: int = 4096) -> bool:
    if not raw:
        return False
    head = raw[:sample]
    if b"\x00" in head:
        return False
    try:
        head.decode("utf-8")
        return True
    except UnicodeDecodeError:
        try:
            head.decode("latin-1")
            return True
        except UnicodeDecodeError:
            return False


def _filename_ext(name: str | None) -> str:
    if not name or "." not in name:
        return ""
    return "." + name.rsplit(".", 1)[1].lower()


async def read_and_validate(
    upload: UploadFile,
    *,
    kinds: Iterable[str],
    max_bytes: int | None = None,
) -> tuple[bytes, str]:
    """
    Reads the upload, enforces size + magic-byte checks, returns (bytes, detected_mime).
    `kinds` is a list of buckets the upload may be: "pdf" | "image" | "text" | "json".
    Raises HTTPException(413|415|422) on failure.
    """
    kinds_set = set(kinds)
    if not kinds_set:
        raise ValueError("kinds must not be empty")

    if max_bytes is None:
        if "image" in kinds_set and len(kinds_set) == 1:
            max_bytes = settings.max_image_bytes
        else:
            max_bytes = settings.max_upload_bytes

    raw = await upload.read()
    size = len(raw)
    if size == 0:
        raise HTTPException(status_code=422, detail="Empty upload")
    if size > max_bytes:
        mb = max_bytes // (1024 * 1024)
        raise HTTPException(
            status_code=413,
            detail=f"Upload exceeds {mb} MB (got {size // (1024 * 1024)} MB)",
        )

    kind = filetype.guess(raw)
    detected_mime = kind.mime if kind else None
    declared_mime = (upload.content_type or "").lower()
    ext = _filename_ext(upload.filename)

    allowed_mimes: set[str] = set()
    for k in kinds_set:
        allowed_mimes.update(_KIND_MIMES.get(k, set()))

    if detected_mime and detected_mime in allowed_mimes:
        return raw, detected_mime

    if "text" in kinds_set or "json" in kinds_set:
        if _looks_like_text(raw):
            if "json" in kinds_set and ext == ".json":
                return raw, "application/json"
            if ext in _TEXT_FALLBACK_EXTS or declared_mime.startswith("text/"):
                return raw, declared_mime or "text/plain"

    if detected_mime:
        raise HTTPException(
            status_code=415,
            detail=(
                f"File content does not match expected types ({sorted(kinds_set)}). "
                f"Detected: {detected_mime}"
            ),
        )

    raise HTTPException(
        status_code=415,
        detail=(
            f"File content type could not be verified for kinds {sorted(kinds_set)}. "
            f"Declared MIME: {declared_mime or 'unset'}, extension: {ext or 'none'}"
        ),
    )


def assert_pdf_page_limit(raw: bytes, *, max_pages: int | None = None) -> int:
    """Returns page count. Raises 422 if PDF parse fails or exceeds page cap."""
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    cap = max_pages if max_pages is not None else settings.max_pdf_pages
    try:
        reader = PdfReader(BytesIO(raw))
        n = len(reader.pages)
    except (PdfReadError, Exception) as exc:
        raise HTTPException(status_code=422, detail=f"Invalid PDF: {exc}") from exc

    if n > cap:
        raise HTTPException(
            status_code=413,
            detail=f"PDF has {n} pages, exceeds limit of {cap}",
        )
    return n
