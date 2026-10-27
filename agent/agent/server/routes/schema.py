"""agent.server.routes.schema: topological diagnosis of database schemas."""

from __future__ import annotations

from fastapi import APIRouter, Body, HTTPException

from agent import schema_complex as sc

router = APIRouter(prefix="/v1/schema")


def _parse_schema(body: dict):
    """Parse a schema model from the request body (shared by analyze/strain)."""
    if body.get("ddl"):
        return sc.parse_schema_ddl(body["ddl"], dialect=body.get("dialect"))
    if body.get("spec"):
        return sc.parse_schema_json(body["spec"])
    if body.get("mongo"):
        return sc.infer_mongo_schema(body["mongo"])
    if body.get("connection") or body.get("mongo_connection"):
        raise HTTPException(400, "Load an offline schema declaration; Agent uses RCDB records")
    raise HTTPException(400, "Provide ddl, spec or an in-memory mongo structure")


@router.post("/analyze")
async def analyze_schema(body: dict = Body(...)):
    """Diagnose a schema's actual topology.

    Provide one of:
      * ``ddl``        - CREATE TABLE / FOREIGN KEY SQL text
      * ``spec``       - JSON schema {tables:[{name,columns,primary_key,foreign_keys}]}
      * ``mongo``      - an in-memory document structure

    Optional: ``store_id`` (persist the schema complex in the RCDB) and
    ``tags``. Returns circular dependency chains, the Hodge hierarchy/
    tension split, implied missing relations (voids), and hub tables.
    """
    try:
        model = _parse_schema(body)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"Could not read schema: {e}") from e

    report = sc.diagnose_schema(model)

    if body.get("lineage_id"):
        try:
            rex, meta = sc.schema_to_rex(model)
            if rex is not None:
                from agent.rcdb import default_store as _store
                from agent.rcdb import version_if_changed
                info = version_if_changed(_store(), body["lineage_id"], rex, meta=meta,
                                          tags=(body.get("tags") or []) + ["schema"])
                report["version"] = info
        except Exception as e:
            report["store_error"] = str(e)
    elif body.get("store_id"):
        try:
            rex, meta = sc.schema_to_rex(model)
            if rex is not None:
                from agent.rcdb import default_store as _store
                _store().put(body["store_id"], rex, meta=meta,
                             tags=(body.get("tags") or []) + ["schema"])
                report["stored_as"] = body["store_id"]
        except Exception as e:
            report["store_error"] = str(e)

    return report


@router.post("/lint")
async def lint_schema(body: dict = Body(...)):
    """Data model lint: per relation character + anomalies + conflict tables."""
    try:
        model = _parse_schema(body)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"Could not read schema: {e}") from e
    return sc.relation_lint(model)


@router.post("/faces")
async def schema_faces_route(body: dict = Body(...)):
    """Explore how the face selection algorithm changes the schema's geometry.

    Same tables & foreign keys, different definition of "what counts as a
    co participation" -> different curl/harmonic reading. Returns, per mode
    ('coparticipation', 'autoface', 'promote', 'none'), the face count, Betti
    numbers, and the Hodge split: a side by side of the schema's topological
    options (e.g. an FK triangle is persistent harmonic under 'coparticipation'
    but bounded curl under 'autoface').
    """
    try:
        model = _parse_schema(body)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"Could not read schema: {e}") from e
    return {"face_options": sc.explore_schema_faces(model),
            "selections": list(sc.SCHEMA_FACE_SELECTIONS)}


@router.post("/strain")
async def schema_strain_route(body: dict = Body(...)):
    """Data forced strain: weight the schema by real data magnitudes and
    measure the geometric strain the data imposes.

    Body: an offline schema (ddl/spec/mongo) with explicit
    ``weights`` {"from->to": cardinality}. Returns the heat map (how much / where), per relation
    attribution (who / what), and effective root causes (how many / coupled).
    """
    try:
        model = _parse_schema(body)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"Could not read schema: {e}") from e

    weights = body.get("weights")
    if body.get("connection"):
        raise HTTPException(400, "Supply declared weights from RCDB; live SQL statistics are not supported")

    result = sc.schema_strain(model, weights=weights)
    return result
