"""Shared connector operations for CLI and HTTP callers.

list_connectors and driver_status report available schemes and drivers.
read builds a complex; validate checks a connector; ingest publishes the result
to an RCStore. Live URIs supply their source by default. In memory schemes
require an explicit source value.
"""

from __future__ import annotations

from typing import Any

from . import _SCHEME_MAP, open_connector, to_rexgraph
from .validate import ValidationReport, validate_connector

# Native files retain their declared fields without a separate cardinality pull.
from agent.record_access import SQL_SCHEMES
WEIGHTABLE_SCHEMES = SQL_SCHEMES


def weight_kwargs(uri: str, want_weights: bool) -> dict[str, Any]:
    """`{'with_weights': True}` only when weights are requested *and* the scheme
    supports them; otherwise empty (so other connectors aren't handed an
    argument their constructor doesn't accept)."""
    return ({"with_weights": True}
            if want_weights and _scheme_of(uri) in WEIGHTABLE_SCHEMES else {})
_DRIVER_PROBE = {
    'mongodb': ('pymongo', 'pip install pymongo'),
    'neo4j': ('neo4j', 'pip install neo4j  (or pass an in-memory structure)'),
    'bolt': ('neo4j', 'pip install neo4j  (or pass an in-memory structure)'),
    'kafka': ('confluent_kafka', 'pip install confluent-kafka  (or pass a structure)'),
    'pulsar': ('pulsar', 'pip install pulsar-client  (or pass a structure)'),
    'ontology': (None, ''),
    'rdf': (None, ''),
    'owl': (None, ''),
    'edges': (None, ''),
    'table': (None, ''),
    'rcdb': (None, ""),
    'rcbd': (None, ""),
    'rcbf': (None, ""),
}

for _scheme in SQL_SCHEMES:
    _DRIVER_PROBE[_scheme] = ("sqlalchemy", "pip install 'rexgraph-agent[sql]' and the source dialect driver")


def _scheme_of(uri: str) -> str:
    scheme = uri.split("://", 1)[0] if "://" in uri else uri
    return scheme.split("+", 1)[0].lower()


def driver_status(scheme: str) -> dict[str, Any]:
    """Whether the driver for ``scheme`` is importable, plus a pip hint if not.
    In memory shapes report available=True (no driver needed)."""
    if scheme.split("+",1)[0].lower() in SQL_SCHEMES:
        try:
            from sqlalchemy.dialects import registry
            dialect=registry.load(scheme.replace("+","."))
            dialect.import_dbapi()
            return {"available":True,"hint":""}
        except Exception:
            return {"available":False,"hint":"pip install 'rexgraph-agent[sql]' and the source dialect driver"}
    scheme = scheme.lower()
    probe, hint = _DRIVER_PROBE.get(scheme, ("", ""))
    if probe is None:                       # in memory shape, no driver
        return {"available": True, "hint": ""}
    if not probe:                           # unknown scheme
        return {"available": False, "hint": ""}
    try:
        __import__(probe)
        return {"available": True, "hint": ""}
    except Exception:
        return {"available": False, "hint": hint}


def list_connectors() -> list[dict[str, Any]]:
    """The registry, grouped by connector: schemes served, advertised
    capabilities, and driver availability for each scheme."""
    by_conn: dict[tuple[str, str], dict[str, Any]] = {}
    for scheme, (mod, cls) in sorted(_SCHEME_MAP.items()):
        key = (mod, cls)
        if key not in by_conn:
            inst = open_connector(scheme)     # scheme alone selects the connector
            caps = inst.capabilities()
            by_conn[key] = {
                "connector": cls,
                "module": f"agent.connectors.{mod}",
                "schemes": [],
                "capabilities": caps.summary(),
                "supports": {
                    "topology": True, "weights": caps.weights,
                    "modality": caps.modality, "faces": caps.faces,
                },
            }
        st = driver_status(scheme)
        by_conn[key]["schemes"].append({
            "scheme": scheme, "driver_available": st["available"],
            "driver_hint": st["hint"],
        })
    return list(by_conn.values())


def _summary(g: Any, meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": meta.get("source"),
        "nV": g.nV, "nE": g.nE, "nF": g.nF,
        "betti": list(g.betti),
        "chain_valid": bool(g.chain_valid),
        "weighted": meta.get("weights") is not None,
        "modality": meta.get("modality") is not None,
    }


def read(uri: str, source: Any = None, **kwargs) -> dict[str, Any]:
    """Build the complex read only and return a summary (no storage). For live
    URI sources, ``source`` defaults to the URI itself."""
    connector = open_connector(uri, **kwargs)
    rex, meta = connector.read(uri if source is None else source)
    from .records import NativeFileConnector, RCDBConnector
    g = rex if isinstance(connector, (NativeFileConnector, RCDBConnector)) else to_rexgraph(rex, meta)
    return _summary(g, meta)


def validate(uri: str, source: Any = None, store_uri: str = "memory://",
             **kwargs) -> ValidationReport:
    """Run the validation harness for ``uri``'s connector against the source."""
    connector = open_connector(uri, **kwargs)
    return validate_connector(connector, uri if source is None else source,
                              store_uri=store_uri)


def ingest(uri: str, record_id: str, *, store: Any = None,
           store_uri: str | None = None, source: Any = None,
           tags: list[str] | None = None, **kwargs) -> dict[str, Any]:
    """Build the complex and persist its *structure* into an RCStore. The only
    writing operation, and it writes solely to the host's own store: either a
    pre opened ``store`` (e.g. the app's singleton) or one opened from
    ``store_uri``."""
    from agent.rcdb import copy_record, open_store
    from agent.record_access import require_store
    from .records import NativeFileConnector, RCDBConnector, _complex

    # Close only what we opened. A caller that handed us its own store, the app's
    # singleton for instance, still owns it and must get it back open.
    ours = None
    if store is None:
        if not store_uri:
            raise ValueError("ingest needs a 'store' object or a 'store_uri'")
        store = ours = open_store(store_uri)

    try:
        require_store(store)
        connector = open_connector(uri, **kwargs)
        if isinstance(connector, RCDBConnector):
            session, selected = connector.select(source)
            graph, meta = _complex(selected.value, "rcdb:" + selected.record.id)
            record = copy_record(session.store, store, selected.record,
                destination_id=record_id, tags=tags, expected_digest=selected.state_digest)
            if record is None:
                raise ValueError("selected RCDB source disappeared")
            return {**_summary(graph, meta), "stored_as": record.id,
                    "version": record.version, "store": store_uri or type(store).__name__}
        rex, meta = connector.read(uri if source is None else source)
        g = rex if isinstance(connector, NativeFileConnector) else to_rexgraph(rex, meta)
        store.put(record_id, g, meta=getattr(g, "_agent_meta", None),
                  tags=list(tags or []))
        out = _summary(g, meta)
        out.update({"stored_as": record_id, "store": store_uri or type(store).__name__})
        return out
    finally:
        if ours is not None:
            ours.close()
