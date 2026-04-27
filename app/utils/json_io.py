import json
import re


_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


def extract_json(text: str) -> dict | list:
    """Parse JSON from an LLM output; tolerate code fences and stray prose."""
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    m = _FENCE_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except Exception:
            pass

    # Fallback: grab the outermost JSON-looking blob.
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end != -1 and end > start:
            candidate = text[start : end + 1]
            try:
                return json.loads(candidate)
            except Exception:
                continue

    raise ValueError("Unable to parse JSON from LLM response")
