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

    first_struct = next((ch for ch in text if ch in "{["), None)
    if first_struct == "[":
        order = (("[", "]"), ("{", "}"))
    else:
        order = (("{", "}"), ("[", "]"))

    candidates: list[str] = []
    for i, (opener, closer) in enumerate(order):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end != -1 and end > start:
            candidate = text[start : end + 1]
            candidates.append(candidate)
            if i == 0 or first_struct is None:
                try:
                    return json.loads(candidate)
                except Exception:
                    continue

    try:
        from json_repair import repair_json
    except ImportError:
        raise ValueError("Unable to parse JSON from LLM response")

    for candidate in [*candidates, text]:
        try:
            repaired = repair_json(candidate, return_objects=True)
            if isinstance(repaired, (dict, list)) and repaired:
                return repaired
        except Exception:
            continue

    raise ValueError("Unable to parse JSON from LLM response")
