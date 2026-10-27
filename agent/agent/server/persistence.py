"""Workspace documents and complete observations persisted in RCDB and RCBD."""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

from ..formats import BUNDLE_SUFFIX, BUNDLE_SUFFIXES, is_bundle_suffix

logger = logging.getLogger(__name__)

def _base_dir() -> Path:
    """Where workspaces live, read when asked rather than captured at import.

    Captured at import this disagreed with `handles.config_dir()`, which reads the
    environment on each call, so one process could hold two different ideas of where the
    configuration is and a containment test would compare paths from different roots.
    One source of truth instead.
    """
    from agent.server.handles import config_dir
    return config_dir() / "workspaces"


def _ws_dir(workspace: str) -> Path:
    """The directory holding one workspace's files.

    The name is validated here because this function both joins it onto a path and
    creates what it names: /api/v1/pipeline/stream took the workspace from a form field,
    so "../.." escaped the workspace root and mkdir made the directories on the way out.
    The predicate is the one `handles` already uses, not a second rule that can drift.
    """
    from agent.server.handles import valid_workspace
    if not valid_workspace(workspace):
        raise ValueError(f"not a usable workspace name: {workspace!r}")
    d = _base_dir() / workspace
    d.mkdir(parents=True, exist_ok=True)
    return d


def _docs_dir(workspace: str) -> Path:
    d = _ws_dir(workspace) / "documents"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _convs_dir(workspace: str) -> Path:
    d = _ws_dir(workspace) / "conversations"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _db_path(workspace: str) -> str:
    return str(_ws_dir(workspace) / "state.rcdb")


def _state_store(workspace):
    from rcdb import LocalStore
    from rcql import register_result_storage_codec
    register_result_storage_codec()
    return LocalStore(_db_path(workspace))


def _write_result(workspace, record_id, value, *, tags=()):
    from contextlib import closing
    from rcql import Result
    from agent.relational import RelationalSession
    from rexgraph.io.publication import publication_lock
    with publication_lock(_ws_dir(workspace)), closing(_state_store(workspace)) as store:
        current = store.get_record(record_id)
        return RelationalSession(store).save_result(record_id, Result((value,)),
            expected_version=0 if current is None else current.version,
            meta={"workspace": workspace}, tags=list(tags))


def _read_result(workspace, record_id):
    from contextlib import closing
    if not Path(_db_path(workspace)).exists():
        return None
    with closing(_state_store(workspace)) as store:
        snapshot = store.read_record(record_id)
        return None if snapshot is None else snapshot.value.values[0]



# Document <-> Session linkage
#
# Sessions (~/.rexgraph-agent/sessions) and workspaces
# (~/.config/rexgraph/workspaces) were separate stores with no
# cross reference. This index makes the mapping bidirectional so a
# workspace document can find its analysis session and vice versa.

def _links_path(workspace: str) -> Path:
    return _ws_dir(workspace) / "doc_sessions.json"


def _settings_path(workspace: str) -> Path:
    return _ws_dir(workspace) / "settings.json"


def load_settings(workspace: str) -> dict:
    """Workspace settings. Absent or unreadable reads as no settings, so a missing
    file never turns a feature on."""
    p = _settings_path(workspace)
    if not p.exists():
        return {}
    try:
        d = json.loads(p.read_text())
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def save_settings(workspace: str, settings: dict) -> dict:
    """Replace a workspace's settings."""
    from rexgraph.io.publication import publication_lock, staged_publication
    path = _settings_path(workspace)
    with publication_lock(path.parent), staged_publication(path) as staged:
        staged.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    return settings


def update_settings(workspace: str, changes: dict) -> dict:
    """Merge changes into a workspace's settings and return the result."""
    from rexgraph.io.publication import publication_lock
    with publication_lock(_ws_dir(workspace)):
        s = load_settings(workspace)
        s.update(changes or {})
        return save_settings(workspace, s)


def link_doc_session(workspace: str, doc_id: str, session_id: str) -> None:
    """Record ``doc_id -> session_id`` for a workspace."""
    from rexgraph.io.publication import publication_lock, staged_publication
    if not doc_id or not session_id:
        return
    p = _links_path(workspace)
    with publication_lock(p.parent):
        data = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        if not isinstance(data, dict):
            raise ValueError("Document session map must be a JSON object")
        data[doc_id] = session_id
        with staged_publication(p) as staged:
            staged.write_text(json.dumps(data, indent=2), encoding="utf-8")


def get_doc_session(workspace: str, doc_id: str):
    """Return the session_id linked to ``doc_id`` (or None)."""
    p = _links_path(workspace)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text()).get(doc_id)
    except Exception:
        return None


def doc_session_map(workspace: str) -> dict:
    """Return the full ``{doc_id: session_id}`` map for a workspace."""
    p = _links_path(workspace)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


# RexGraph persistence (uses rexgraph/io/bundle.py directly)

def staging_dir(workspace: str) -> Path:
    """Where a file this workspace uploaded is kept while a document refers to it.

    Not the shared temp directory. A staged upload becomes the document's source path,
    so it deliberately outlives the request, and gettempdir() is both nameable by anyone
    who may name a path and walked by the routes that accept a directory, so one
    tenant's upload sat next to another's indefinitely and enumerating it needed no
    guessed filename. Under the workspace instead, which no caller can name at all
    because path_within refuses the deployment's config directory outright.
    """
    d = _ws_dir(workspace) / "staging"
    d.mkdir(parents=True, exist_ok=True)
    return d


def doc_path(workspace: str, doc_id: str, suffix: str = BUNDLE_SUFFIX) -> Path:
    """The path of one document, with the id held inside its own workspace.

    Validating the workspace name settled which directory this is; it said nothing about
    what gets joined onto it. A doc_id of "../../victim/documents/ordinary" resolved into
    another workspace, and RexBundle.save creates parents, so a tenant could plant or
    overwrite a document in a workspace they cannot otherwise reach. Reproduced, not
    theorised.

    `handles.path_within` is the wrong predicate here and deliberately so: it refuses
    anything under the deployment's config directory, which is exactly where every
    workspace lives. That predicate answers whether a caller may name a path on the
    filesystem at all. This one answers whether an id stays inside its own workspace.
    """
    base = _docs_dir(workspace).resolve()
    p = (base / f"{doc_id}{suffix}").resolve()
    if p == base or base not in p.parents:
        raise ValueError(f"not a document id: {doc_id!r}")
    return p


def existing_doc_path(workspace: str, doc_id: str) -> Path | None:
    """Where a document actually is, canonical suffix first and legacy after.

    Writes go to `BUNDLE_SUFFIX` alone, so a document saved before the rename is only
    reachable by asking for its old suffix too. Ordering is not cosmetic: while both
    exist for one id, the canonical one is the newer and must win.
    """
    for suffix in BUNDLE_SUFFIXES:
        candidate = doc_path(workspace, doc_id, suffix)
        if candidate.exists():
            return candidate
    return None


def save_document_rex(workspace: str, doc_id: str, rex, cache="all"):
    """Save a document's RexGraph as an RCBD bundle."""
    from rexgraph.io import save_rcbd
    path = str(doc_path(workspace, doc_id))
    save_rcbd(path, rex, cache=cache)
    return path


def load_document_rex(workspace: str, doc_id: str):
    """Load a document's RexGraph from its bundle, canonical or legacy."""
    from rexgraph.io import load_rcbd
    path = existing_doc_path(workspace, doc_id)
    if path is None:
        return None
    return load_rcbd(str(path))


def list_document_bundles(workspace: str) -> list[str]:
    """List all saved document IDs in a workspace."""
    d = _docs_dir(workspace)
    # A stem can appear under both suffixes mid migration; it is still one document.
    return sorted({
        p.stem for p in d.iterdir()
        if is_bundle_suffix(p.suffix) or p.is_dir()
    })


# Analysis records

def save_analysis(workspace: str, doc_id: str, rex, analysis: dict):
    """Persist the complete graph and analysis in one canonical RCQL result."""
    return _write_result(workspace, "analysis:" + doc_id, {"source": rex, "analysis": analysis})


def load_analysis(workspace: str, doc_id: str):
    return _read_result(workspace, "analysis:" + doc_id)


def save_analysis_sql(workspace: str, doc_id: str, rex, analysis: dict):
    """Compatibility entry point for RCDB analysis publication."""
    return save_analysis(workspace, doc_id, rex, analysis)



# Activity persistence

def save_activity(workspace: str, activity_edges: list):
    """Publish the complete current activity snapshot as an RCDB revision."""
    return _write_result(workspace, "activity", list(activity_edges))



def load_activity(workspace: str) -> list:
    """Read the current complete activity snapshot."""
    value = _read_result(workspace, "activity")
    if value is not None:
        return value
    path = _ws_dir(workspace) / "activity.json"
    if path.exists():
        data = json.loads(path.read_text())
        return [(d["user"], d["action"], d["target"], d["ts"]) for d in data]
    return []



# Query history persistence

def save_query(workspace: str, user_id: str, query_text: str,
               mode: str, results: list):
    """Publish one complete query observation with an independent identity."""
    import uuid
    value = {"user": user_id, "query": query_text, "mode": mode,
             "results": results, "timestamp": time.time()}
    return _write_result(workspace, "query:" + uuid.uuid4().hex, value, tags=("agent-query",))



def load_query_history(workspace: str, limit: int = 50) -> list:
    """Read recent query records through typed RCQL selection."""
    from contextlib import closing
    from agent.record_access import read_query
    if type(limit) is not int or not 0 <= limit <= 10000:
        raise ValueError("query history limit must be in 0..10000")
    if not limit or not Path(_db_path(workspace)).exists():
        return []
    with closing(_state_store(workspace)) as store:
        selected = read_query(store, 'FROM $db RETURN RCDB_SELECT($filters,$limit)',
            parameters={"filters": {"tags_all": ["agent-query"]}, "limit": limit}).values[0]
        values = [store.read_record(row["id"], version=row["version"]).value.values[0] for row in selected]
    return sorted(values, key=lambda value: value["timestamp"], reverse=True)



# Conversation persistence

def save_conversation(workspace: str, session_id: str, exchanges: list):
    """Publish complete recorded exchange measurements in RCDB."""
    data = [{"n_shared": ex.n_shared, "exchange_edges": ex.n_exchange_edges,
             "kappa": ex.kappa_mean,
             "hodge": [ex.hodge_gradient, ex.hodge_curl, ex.hodge_harmonic]} for ex in exchanges]
    return _write_result(workspace, "conversation:" + session_id, data)


def load_conversation(workspace: str, session_id: str):
    return _read_result(workspace, "conversation:" + session_id)



# Export

def export_workspace(workspace: str, output_path: str, fmt: str = "rex"):
    """Export an RCDB workspace backup or a JSON observation summary."""
    ws_dir = _ws_dir(workspace)
    if fmt == "json":
        from rexgraph.io._compat import dumps
        export = {"workspace": workspace, "documents": list_document_bundles(workspace),
                  "activity": load_activity(workspace), "queries": load_query_history(workspace, limit=1000)}
        Path(output_path).write_text(dumps(export, nan="null"))
        return output_path
    if fmt in {"rex", "rcdb"}:
        import shutil
        from contextlib import closing, nullcontext
        from rexgraph.io.publication import publication_lock
        target = Path(output_path).resolve()
        if target == ws_dir.resolve() or ws_dir.resolve() in target.parents:
            raise ValueError("workspace backup must be outside its source")
        if target.exists():
            raise FileExistsError(target)
        with publication_lock(ws_dir):
            scope = closing(_state_store(workspace)) if Path(_db_path(workspace)).exists() else nullcontext(None)
            with scope as store:
                lock = store.write_scope() if store is not None else nullcontext()
                with lock:
                    shutil.copytree(ws_dir, target, ignore=shutil.ignore_patterns("state.db", "state.db-*"))
        return output_path
    raise ValueError(f"Unknown format: {fmt}")



# Shared file browsing

def list_workspace_files(workspace: str) -> list:
    """List all files in a workspace with metadata.

    Returns info about each document: doc_id, file size, format,
    when it was saved, whether analysis exists.
    """
    docs_dir = _docs_dir(workspace)
    from contextlib import closing, nullcontext
    scope = closing(_state_store(workspace)) if Path(_db_path(workspace)).exists() else nullcontext(None)
    with scope as store:
        return _workspace_files(docs_dir, store)


def _workspace_files(docs_dir, store):
    files = []
    for entry in sorted(docs_dir.iterdir()):
        info = {"name": entry.name, "doc_id": entry.stem}

        if entry.is_dir() and (entry / "MANIFEST.json").exists():
            # A directory bundle, RCBD or legacy: the manifest identifies it, not the
            # suffix, so this branch is already correct for both. "rex" here is the
            # export FORMAT KEY, which is deliberately unchanged by the rename.
            info["format"] = "rex"
            try:
                manifest = json.loads((entry / "MANIFEST.json").read_text())
                info["nV"] = manifest.get("nV", 0)
                info["nE"] = manifest.get("nE", 0)
                info["nF"] = manifest.get("nF", 0)
            except Exception:
                pass
            info["size_bytes"] = sum(
                f.stat().st_size for f in entry.rglob("*") if f.is_file()
            )
            info["modified"] = entry.stat().st_mtime
        elif is_bundle_suffix(entry.suffix):
            info["format"] = "rex"
            info["size_bytes"] = entry.stat().st_size
            info["modified"] = entry.stat().st_mtime
        else:
            continue

        # Check the native workspace store
        info["has_analysis"] = (store is not None and
            store.get_record("analysis:" + entry.stem) is not None)

        files.append(info)

    return files


def get_workspace_stats(workspace: str) -> dict:
    """Get aggregate statistics for a workspace."""
    files = list_workspace_files(workspace)
    queries = load_query_history(workspace, limit=10000)
    activity = load_activity(workspace)

    users = set()
    for user, _action, _target, _ts in activity:
        users.add(user)

    return {
        "workspace": workspace,
        "n_documents": len(files),
        "total_size_bytes": sum(f.get("size_bytes", 0) for f in files),
        "n_queries": len(queries),
        "n_users": len(users),
        "users": sorted(users),
        "has_database": os.path.exists(_db_path(workspace)),
    }
