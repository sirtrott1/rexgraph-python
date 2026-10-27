"""Immutable text selections with explicit source positions and emitted cuts."""
from dataclasses import dataclass
import hashlib


@dataclass(frozen=True)
class ContextSelection:
    text: str
    parts: tuple
    limit: int
    truncated: bool

    def as_record(self):
        return {"text": self.text, "parts": [dict(part) for part in self.parts],
                "character_limit": self.limit, "truncated": self.truncated,
                "encoding": "utf-8", "sha256": hashlib.sha256(self.text.encode("utf-8")).hexdigest()}


def select_context(entries, *, limit=12000):
    """Select ordered chunk uses without identifying equal text from different sources."""
    if type(limit) is not int or limit < 0:
        raise ValueError("context limit requires a nonnegative integer")
    text, parts, truncated = "", [], False
    for doc, chunk in entries:
        separator = "\n\n" if text else ""
        remaining = limit-len(text)-len(separator)
        if remaining <= 0:
            truncated = True
            break
        emitted = chunk.text[:remaining]
        start, end = getattr(chunk, "char_start", None), getattr(chunk, "char_end", None)
        source_text = getattr(doc, "text", None)
        mapped = (type(start) is int and type(end) is int and isinstance(source_text, str)
                  and 0 <= start <= end <= len(source_text) and source_text[start:end] == chunk.text)
        digest = None
        if getattr(doc, "rex", None) is not None:
            from rexgraph.object_identity import object_digest
            digest = object_digest(doc.rex)
        part = {"doc_id": str(doc.doc_id), "source_state_digest": digest,
                "source_text_sha256": hashlib.sha256(source_text.encode("utf-8")).hexdigest()
                    if isinstance(source_text, str) else None,
                "context_start": len(text)+len(separator), "context_end": len(text)+len(separator)+len(emitted),
                "source_char_start": start if mapped else None,
                "source_char_end": start+len(emitted) if mapped else None,
                "source_byte_start": len(source_text[:start].encode("utf-8")) if mapped else None,
                "source_byte_end": len(source_text[:start+len(emitted)].encode("utf-8")) if mapped else None,
                "mapping_status": "mapped" if mapped else "unmapped"}
        metadata = getattr(doc, "meta", {}) or {}
        for name in ("record_id", "record_version", "source_cutoff"):
            value = metadata.get(name)
            if value is not None:
                if type(value) not in (int, float, str, bool, bytes):
                    raise TypeError("context source references require portable scalar metadata")
                part[name] = value
        parts.append(tuple(part.items()))
        text += separator+emitted
        if len(emitted) != len(chunk.text):
            truncated = True
            break
    return ContextSelection(text, tuple(parts), limit, truncated)


def diagnostic_source(selection):
    """Build the declared text diagnostic on exactly the emitted context."""
    from agent.adapters.text import TextAdapter
    from agent.auto import build_rex_from_edges
    edges = TextAdapter().build(selection.text, min_count=1, max_vocab=10000)
    return (build_rex_from_edges(edges), edges.vertex_labels) if edges.nE else (None, edges.vertex_labels)
