import re
import uuid

from app.core.config import settings
from app.models.chunk import Chunk


_WHITESPACE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip()


def chunk_text(
    text: str,
    *,
    doc_id: str,
    metadata: dict | None = None,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[Chunk]:
    size = chunk_size or settings.chunk_size
    olap = overlap if overlap is not None else settings.chunk_overlap
    text = _normalize(text)
    if not text:
        return []

    chunks: list[Chunk] = []
    start = 0
    idx = 0
    meta_base = dict(metadata or {})

    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            space = text.rfind(" ", start, end)
            if space > start + int(size * 0.5):
                end = space
        piece = text[start:end].strip()
        if piece:
            cmeta = {**meta_base, "doc_id": doc_id, "chunk_index": idx}
            chunks.append(
                Chunk(
                    id=f"{doc_id}:{idx}:{uuid.uuid4().hex[:8]}",
                    text=piece,
                    metadata=cmeta,
                )
            )
            idx += 1
        if end >= len(text):
            break
        start = max(0, end - olap)
    return chunks
