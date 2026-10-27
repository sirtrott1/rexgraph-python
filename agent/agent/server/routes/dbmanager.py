"""RCDB record and RCQL execution views over the existing Agent tool boundary."""
from fastapi import APIRouter, Body, Depends, HTTPException, Request
from agent.server.auth import TokenEntry, require_auth

router = APIRouter(prefix="/v1/dbmanager")


async def _call(request, token, name, arguments):
    from .rex import call_tool
    return await call_tool(request, {"name": name, "arguments": arguments}, token)


@router.get("/records")
async def records(request: Request, limit: int = 100, offset: int = 0,
                  token: TokenEntry = Depends(require_auth)):
    return await _call(request, token, "rexgraph_rcdb_list", {"limit": limit, "offset": offset})


@router.post("/query")
async def query(request: Request, body: dict = Body(...), token: TokenEntry = Depends(require_auth)):
    return await _call(request, token, "rexgraph_rcql_query", body)


@router.post("/inspect")
async def inspect_record(request: Request, body: dict = Body(...), token: TokenEntry = Depends(require_auth)):
    return await _call(request, token, "rexgraph_rcdb_inspect", body)


@router.post("/publish")
async def publish(request: Request, body: dict = Body(...), token: TokenEntry = Depends(require_auth)):
    return await _call(request, token, "rexgraph_rcdb_publish", body)


@router.post("/import")
async def import_record(request: Request, body: dict = Body(...), token: TokenEntry = Depends(require_auth)):
    return await _call(request, token, "rexgraph_rcdb_import_file", body)


@router.get("/connections")
async def connections(token: TokenEntry = Depends(require_auth)):
    from agent.server.scope import secret_store
    return {"connections":[{"name":row["name"],"kind":row.get("kind","sql")}
                           for row in secret_store().list()],"provider":"rcdb"}


@router.post("/connections")
async def save_connection(body: dict = Body(...), token: TokenEntry = Depends(require_auth)):
    from agent.server.scope import secret_store
    from agent.server.dbguard import check_db_uri
    from agent.record_access import SQL_SCHEMES
    name,uri=body.get("name"),body.get("uri")
    if not isinstance(name,str) or not name.strip() or not isinstance(uri,str):
        raise HTTPException(400,"provide a connection name and SQL URI")
    if uri.split(":",1)[0].split("+",1)[0].lower() not in SQL_SCHEMES:
        raise HTTPException(400,"select a SQL source dialect")
    check_db_uri(uri,allow_sql=True)
    secret_store().put(name.strip(),uri,"sql")
    return {"saved":name.strip(),"primary_store":"rcdb"}


@router.delete("/connections/{name}")
async def delete_connection(name: str, token: TokenEntry = Depends(require_auth)):
    from agent.server.scope import secret_store
    return {"deleted":secret_store().delete(name)}


@router.post("/sql/catalog")
@router.post("/tables")
@router.post("/test")
async def sql_catalog(request: Request, body: dict = Body(...), token: TokenEntry = Depends(require_auth)):
    if "name" in body:body={**body,"connection":body["name"]};body.pop("name")
    return await _call(request,token,"rexgraph_sql_catalog",body)


@router.post("/sql/import")
async def sql_import(request: Request, body: dict = Body(...), token: TokenEntry = Depends(require_auth)):
    return await _call(request,token,"rexgraph_sql_import",body)


@router.get("/sql/export")
async def sql_export(record_id: str, format: str = "rcbd", version: int | None = None,
                     token: TokenEntry = Depends(require_auth)):
    from agent.server import artifacts
    from agent.sql_source import SQLDataset
    from agent.rcdb import default_store
    dataset=SQLDataset(default_store(),record_id,version=version)
    from starlette.concurrency import run_in_threadpool
    return await run_in_threadpool(artifacts.download,lambda path:dataset.export(path,format=format),".sql-import","rcdb-import.zip")


@router.post("/strain")
async def removed_strain(body: dict = Body(...)):
    raise HTTPException(410,"Use offline schema weights or import SQL data into RCDB")


@router.post("/ddl")
async def generate_ddl(body: dict = Body(...)):
    """Render an offline schema declaration without executing it."""
    from agent import schema_complex as sc
    if not body.get("spec"):
        raise HTTPException(400, "Provide a schema spec")
    return {"ddl": sc.export_schema_ddl(sc.parse_schema_json(body["spec"]))}
