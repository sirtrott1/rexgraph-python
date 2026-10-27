"""SQL source tools under Agent's existing workspace and file boundaries."""


def source(uri=None, connection=None, files=None, schema=None, _ctx=None):
    if sum(bool(value) for value in (uri,connection,files))!=1:
        raise ValueError("select one SQL URI, saved connection, or uploaded SQLite file")
    if connection:
        from agent.server.scope import secret_store
        selected_secrets=secret_store()
        if _ctx is not None and _ctx.auth_enabled:
            from agent.server.scope import ScopedSecrets
            from agent.secrets import open_secret_store
            selected_secrets=ScopedSecrets(open_secret_store(),_ctx.workspace)
        uri=selected_secrets.get(connection)
    if files:
        if len(files)!=1:raise ValueError("select one uploaded SQLite file")
        try:
            from sqlalchemy.engine import URL
        except ImportError as exc:
            raise ImportError("SQL sources require pip install 'rexgraph-agent[sql]'") from exc
        uri=URL.create("sqlite",database=str(files[0])).render_as_string(hide_password=False)
    if _ctx is not None:
        from agent.server.dbguard import check_db_uri
        from agent.record_access import SQL_SCHEMES
        scheme=uri.split(":",1)[0].split("+",1)[0].lower()
        if scheme not in SQL_SCHEMES:raise ValueError("source must use an installed SQL dialect")
        if scheme in {"sqlite","duckdb","access"} and not files:
            raise ValueError("upload a SQL file handle instead of naming a server file")
        if not files:check_db_uri(uri,allow_sql=True)
    from agent.sql_source import SQLSource
    return SQLSource(uri,schema=schema)


def catalog(uri=None,connection=None,files=None,schema=None,_ctx=None):
    with source(uri,connection,files,schema,_ctx) as selected:
        return {"tables":selected.catalog(),"origin":selected.origin(),"primary_store":"rcdb"}


def import_records(record_id,expected_version,uri=None,connection=None,files=None,schema=None,
                   tables=None,batch_size=1000,_ctx=None):
    from agent.relational import _session
    from agent.sql_source import import_sql
    from agent.server.budget import Deadline,check_size
    deadline=Deadline()
    def progress(value):
        deadline.check("SQL import")
        check_size({"rows":value["rows"]})
    selected=source(uri,connection,files,schema,_ctx)
    return import_sql(selected,_session(_ctx).store,record_id,expected_version=expected_version,
                      tables=tables,batch_size=batch_size,progress=progress)


# Keep driver diagnostics, which may contain credentials, off socket responses.
def _driver_errors(handler):
    from functools import wraps
    @wraps(handler)
    def execute(*args,**kwargs):
        try:return handler(*args,**kwargs)
        except ImportError:
            raise
        except Exception as exc:
            import sqlalchemy.exc
            if isinstance(exc,sqlalchemy.exc.SQLAlchemyError):
                raise ValueError(f"SQL source operation failed ({type(exc).__name__}); check the source driver, access and schema") from exc
            raise
    return execute

catalog=_driver_errors(catalog)
import_records=_driver_errors(import_records)
