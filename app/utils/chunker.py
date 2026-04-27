"""
Structure-aware chunker.

Pipeline:
1. Split the document into structural blocks: headings, paragraphs, list groups,
   tables. Tables are kept atomic regardless of size — splitting them destroys meaning.
2. Group blocks under their nearest heading path (e.g. "Sales > Order > Confirmation").
3. Within each section, run token-aware windowing using tiktoken so chunk_size is
   measured in tokens rather than characters (LLMs see tokens, embedding models bill
   by tokens). Falls back to character windowing if tiktoken is unavailable.
4. Add neighbor pointers (prev/next chunk id) and section metadata so retrieval
   can expand context without a separate parent lookup.

Public API is unchanged:
  chunk_text(text, *, doc_id, metadata=None, chunk_size=None, overlap=None) -> list[Chunk]
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Iterable

from app.core.config import settings
from app.models.chunk import Chunk


_HEADING_MD = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_HEADING_NUMBERED = re.compile(r"^(\d+(?:\.\d+){0,5})[\.\)]?\s+([A-Z][^\n]{2,120})$")
_HEADING_ALLCAPS = re.compile(r"^([A-Z][A-Z0-9 \-_/&]{3,80})$")
_LIST_ITEM = re.compile(r"^\s*(?:[-*•]|\d+[\.\)])\s+\S")
_TABLE_ROW_PIPE = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_SEP = re.compile(r"^\s*\|?[\s\-:|]+\|[\s\-:|]+\|?\s*$")
_BLANK = re.compile(r"^\s*$")


@dataclass
class _Block:
    kind: str  # "heading" | "paragraph" | "list" | "table"
    text: str
    level: int = 0  # heading level; 0 for non-headings
    headings: list[tuple[int, str]] = field(default_factory=list)  # snapshot at block time


def _classify_line(line: str) -> tuple[str, int, str]:
    """Returns (kind, level, normalized_text). kind ∈ heading|list|table|para|blank."""
    if _BLANK.match(line):
        return "blank", 0, ""
    if _TABLE_ROW_PIPE.match(line):
        return "table", 0, line.rstrip()

    m = _HEADING_MD.match(line)
    if m:
        return "heading", len(m.group(1)), m.group(2).strip()

    m = _HEADING_NUMBERED.match(line)
    if m:
        depth = m.group(1).count(".") + 1
        return "heading", min(depth, 6), f"{m.group(1)} {m.group(2).strip()}"

    if _HEADING_ALLCAPS.match(line) and len(line.strip()) <= 80:
        return "heading", 2, line.strip()

    if _LIST_ITEM.match(line):
        return "list", 0, line.rstrip()

    return "para", 0, line.rstrip()


def _split_blocks(text: str) -> list[_Block]:
    """Walk the text once, emit structural blocks. Tables and list-groups stay together."""
    blocks: list[_Block] = []
    headings_stack: list[tuple[int, str]] = []

    lines = text.splitlines()
    i = 0
    n = len(lines)
    while i < n:
        kind, level, content = _classify_line(lines[i])

        if kind == "blank":
            i += 1
            continue

        if kind == "heading":
            while headings_stack and headings_stack[-1][0] >= level:
                headings_stack.pop()
            headings_stack.append((level, content))
            blocks.append(
                _Block(
                    kind="heading",
                    text=content,
                    level=level,
                    headings=list(headings_stack),
                )
            )
            i += 1
            continue

        if kind == "table":
            buf: list[str] = []
            while i < n and (_TABLE_ROW_PIPE.match(lines[i]) or _TABLE_SEP.match(lines[i])):
                buf.append(lines[i])
                i += 1
            blocks.append(
                _Block(kind="table", text="\n".join(buf), headings=list(headings_stack))
            )
            continue

        if kind == "list":
            buf = []
            while i < n:
                k2, _, c2 = _classify_line(lines[i])
                if k2 == "list" or (k2 == "para" and buf and lines[i].startswith(("  ", "\t"))):
                    buf.append(c2 if k2 == "list" else lines[i].rstrip())
                    i += 1
                    continue
                break
            blocks.append(
                _Block(kind="list", text="\n".join(buf), headings=list(headings_stack))
            )
            continue

        # paragraph: gather consecutive non-blank, non-structural lines
        buf = [content]
        i += 1
        while i < n:
            k2, _, c2 = _classify_line(lines[i])
            if k2 in {"para"} and not _LIST_ITEM.match(lines[i]):
                buf.append(c2)
                i += 1
                continue
            break
        blocks.append(
            _Block(kind="paragraph", text=" ".join(buf), headings=list(headings_stack))
        )

    return blocks


_TOK = None


def _get_tokenizer():
    global _TOK
    if _TOK is False:
        return None
    if _TOK is not None:
        return _TOK
    try:
        import tiktoken

        _TOK = tiktoken.get_encoding("cl100k_base")
        return _TOK
    except Exception:
        _TOK = False
        return None


def _measure(text: str) -> int:
    tok = _get_tokenizer()
    if tok is not None:
        try:
            return len(tok.encode(text))
        except Exception:
            pass
    return max(1, len(text) // 4)


def _split_oversized_paragraph(text: str, max_units: int, overlap_units: int) -> list[str]:
    """Token-aware sliding window for prose blocks that exceed the budget."""
    tok = _get_tokenizer()
    if tok is None:
        out: list[str] = []
        size = max(50, max_units * 4)
        olap = max(0, overlap_units * 4)
        start = 0
        while start < len(text):
            end = min(start + size, len(text))
            if end < len(text):
                space = text.rfind(" ", start, end)
                if space > start + size // 2:
                    end = space
            piece = text[start:end].strip()
            if piece:
                out.append(piece)
            if end >= len(text):
                break
            start = max(0, end - olap)
        return out

    ids = tok.encode(text)
    out = []
    start = 0
    while start < len(ids):
        end = min(start + max_units, len(ids))
        piece = tok.decode(ids[start:end]).strip()
        if piece:
            out.append(piece)
        if end >= len(ids):
            break
        start = max(0, end - overlap_units)
    return out


def _heading_path(headings: list[tuple[int, str]]) -> str:
    return " > ".join(h[1] for h in headings) if headings else ""


def _pack_section(
    section_blocks: list[_Block],
    *,
    max_units: int,
    overlap_units: int,
) -> Iterable[tuple[str, dict]]:
    """Yields (text, extra_metadata) for each chunk in this section."""
    headings = section_blocks[0].headings if section_blocks else []
    section_title = headings[-1][1] if headings else ""
    heading_path = _heading_path(headings)

    base_meta = {
        "section_title": section_title,
        "heading_path": heading_path,
        "heading_level": headings[-1][0] if headings else 0,
    }

    buf: list[str] = []
    buf_units = 0

    def flush(extra: dict | None = None) -> Iterable[tuple[str, dict]]:
        nonlocal buf, buf_units
        if not buf:
            return
        text = "\n\n".join(buf).strip()
        if text:
            md = {**base_meta}
            if extra:
                md.update(extra)
            yield text, md
        buf = []
        buf_units = 0

    for block in section_blocks:
        if block.kind == "heading":
            continue

        if block.kind == "table":
            yield from flush()
            yield block.text, {**base_meta, "block_kind": "table", "atomic": True}
            continue

        block_units = _measure(block.text)

        if block_units > max_units and block.kind == "paragraph":
            yield from flush()
            for piece in _split_oversized_paragraph(block.text, max_units, overlap_units):
                yield piece, {**base_meta, "block_kind": "paragraph"}
            continue

        if buf_units + block_units > max_units and buf:
            yield from flush()

        buf.append(block.text)
        buf_units += block_units

    yield from flush()


def chunk_text(
    text: str,
    *,
    doc_id: str,
    metadata: dict | None = None,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[Chunk]:
    """
    Structure-aware chunking. Returns list[Chunk] with metadata that includes
    section_title, heading_path, prev_chunk_id, next_chunk_id, chunk_index.
    chunk_size / overlap are interpreted as TOKENS when tiktoken is available,
    otherwise as characters (legacy behaviour).
    """
    if not text or not text.strip():
        return []

    raw_size = chunk_size or settings.chunk_size
    raw_overlap = overlap if overlap is not None else settings.chunk_overlap
    if _get_tokenizer() is not None:
        max_units = max(64, raw_size // 4)
        overlap_units = max(0, raw_overlap // 4)
    else:
        max_units = raw_size
        overlap_units = raw_overlap

    blocks = _split_blocks(text)
    if not blocks:
        return []

    sections: list[list[_Block]] = []
    current: list[_Block] = []
    for b in blocks:
        if b.kind == "heading":
            if current:
                sections.append(current)
            current = [b]
        else:
            current.append(b)
    if current:
        sections.append(current)

    meta_base = dict(metadata or {})
    chunks: list[Chunk] = []
    section_index = 0

    for section_blocks in sections:
        for piece_text, extra_md in _pack_section(
            section_blocks, max_units=max_units, overlap_units=overlap_units
        ):
            idx = len(chunks)
            cmeta = {
                **meta_base,
                "doc_id": doc_id,
                "chunk_index": idx,
                "section_index": section_index,
                **extra_md,
            }
            chunks.append(
                Chunk(
                    id=f"{doc_id}:{idx}:{uuid.uuid4().hex[:8]}",
                    text=piece_text,
                    metadata=cmeta,
                )
            )
        section_index += 1

    for i, c in enumerate(chunks):
        c.metadata["prev_chunk_id"] = chunks[i - 1].id if i > 0 else None
        c.metadata["next_chunk_id"] = chunks[i + 1].id if i + 1 < len(chunks) else None

    return chunks
