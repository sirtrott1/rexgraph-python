"""Optional SQL sources and verified native imports into RCDB."""
from __future__ import annotations

from contextlib import ExitStack, nullcontext
from datetime import date, datetime, time
from decimal import Decimal
from pathlib import Path
import json
import struct
import uuid
from threading import local, RLock, get_ident
from weakref import WeakKeyDictionary

_pool_guard=RLock()
_pool_locks=WeakKeyDictionary()

import numpy as np


def _encode(value):
    if value is None: return ("null", None)
    if type(value) is bool: return ("bool", value)
    if type(value) is int: return ("int", value)
    if type(value) is float: return ("float64", struct.pack(">d", value))
    if isinstance(value, str): return ("text", value)
    if isinstance(value, (bytes, bytearray, memoryview)): return ("bytes", bytes(value))
    if isinstance(value, Decimal): return ("decimal", str(value))
    if isinstance(value, datetime):
        return ("datetime", (value.isoformat(), value.fold, getattr(value.tzinfo, "key", None)))
    if isinstance(value, time): return ("time", (value.isoformat(), value.fold))
    if isinstance(value, date): return ("date", value.isoformat())
    if isinstance(value, uuid.UUID): return ("uuid", value.bytes)
    if isinstance(value, list): return ("list", tuple(_encode(item) for item in value))
    if isinstance(value, tuple): return ("tuple", tuple(_encode(item) for item in value))
    if isinstance(value, dict): return ("map", tuple((_encode(k), _encode(v)) for k,v in value.items()))
    raise TypeError(f"SQL driver value {type(value).__name__} needs an explicit native encoding")


def _decode(value):
    kind, data = value
    if kind in {"null", "bool", "int", "text", "bytes"}: return data
    if kind == "float64": return struct.unpack(">d", data)[0]
    if kind == "decimal": return Decimal(data)
    if kind == "date": return date.fromisoformat(data)
    if kind == "time": return time.fromisoformat(data[0]).replace(fold=data[1])
    if kind == "datetime":
        result = datetime.fromisoformat(data[0]).replace(fold=data[1])
        if data[2] is not None:
            from zoneinfo import ZoneInfo
            candidate = result.replace(tzinfo=ZoneInfo(data[2]))
            if candidate.utcoffset() != result.utcoffset():
                raise ValueError("stored SQL datetime differs from current timezone rules")
            result = candidate
        return result
    if kind == "uuid": return uuid.UUID(bytes=data)
    if kind == "list": return [_decode(item) for item in data]
    if kind == "tuple": return tuple(_decode(item) for item in data)
    if kind == "map": return {_decode(k): _decode(v) for k,v in data}
    raise ValueError("unknown SQL scalar encoding")


def _size(value, name, maximum=100000):
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be an integer in 1..{maximum}")
    return value


class SQLSource:
    """Read catalog declarations and bounded rows using an explicit SQL source.

    SQL statements are constructed from catalog table/column selectors. The
    source never executes caller-supplied SQL or publishes Agent state into SQL.
    """
    def __init__(self, source, *, schema=None, isolation=None):
        try:
            import sqlalchemy as sa
        except ImportError as exc:
            raise ImportError("SQL sources require pip install 'rexgraph-agent[sql]'") from exc
        from rexgraph.io.sql_bridge import get_engine
        self.engine = get_engine(source) if isinstance(source, str) else source
        if not isinstance(self.engine, sa.engine.Engine):
            raise TypeError("SQLSource requires a URI or SQLAlchemy Engine")
        if self.engine.dialect.name == "sqlite":
            database = self.engine.url.database or ""
            if database and ":memory:" not in database and self.engine.url.query.get("mode") != "memory":
                from urllib.parse import urlparse,unquote
                filename=unquote(urlparse(database).path) if database.startswith("file:") else database
                if not Path(filename).is_file():raise FileNotFoundError("SQL source file does not exist")
        self.schema, self.isolation = schema, isolation
        self._connection = self._transaction = None
        self._sessions = local()
        self._thread_owner=get_ident() if isinstance(self.engine.pool,sa.pool.SingletonThreadPool) else None

    def __enter__(self):
        if self._connection is not None: raise RuntimeError("SQL source snapshot is already open")
        if self._thread_owner is not None and self._thread_owner!=get_ident():
            raise RuntimeError("thread-local SQL pools cannot cross worker threads; use a shared source engine")
        import sqlalchemy as sa
        snapshot=object.__new__(type(self));snapshot.__dict__=self.__dict__.copy()
        snapshot._pool_gate=None
        if isinstance(self.engine.pool,sa.pool.StaticPool):
            with _pool_guard:
                snapshot._pool_gate=_pool_locks.setdefault(self.engine,RLock())
            snapshot._pool_gate.acquire()
        try:
            snapshot._connection=self.engine.connect()
            if self.isolation:
                snapshot._connection=snapshot._connection.execution_options(isolation_level=self.isolation)
            snapshot._transaction=snapshot._connection.begin()
            if self.engine.dialect.name == "sqlite":snapshot._connection.exec_driver_sql("BEGIN")
        except BaseException:
            if snapshot._connection is not None:snapshot._connection.close()
            if snapshot._pool_gate is not None:snapshot._pool_gate.release()
            raise
        stack=getattr(self._sessions,"stack",None)
        if stack is None:stack=[];self._sessions.stack=stack
        stack.append(snapshot)
        return snapshot

    def __exit__(self, *args):
        snapshot=self._sessions.stack.pop()
        try:
            if snapshot._transaction is not None:snapshot._transaction.rollback()
        finally:
            try:snapshot._connection.close()
            finally:
                snapshot._connection=snapshot._transaction=None
                if snapshot._pool_gate is not None:snapshot._pool_gate.release()

    def origin(self):
        url = self.engine.url
        return {"dialect": self.engine.dialect.name, "driver": self.engine.dialect.driver,
                "host": url.host, "port": url.port, "database":
                    Path(url.database).name if self.engine.dialect.name=="sqlite" and url.database else url.database,
                "schema": self.schema}

    def catalog(self, tables=None):
        if self._connection is None:
            with self as snapshot: return snapshot.catalog(tables)
        import sqlalchemy as sa
        inspector = sa.inspect(self._connection)
        available = inspector.get_table_names(schema=self.schema)
        selected = available if tables is None else list(tables)
        if len(set(selected)) != len(selected) or any(type(t) is not str or t not in available for t in selected):
            raise ValueError("select distinct existing catalog table names")
        result = []
        for name in selected:
            columns = inspector.get_columns(name, schema=self.schema)
            row = {"table": name, "schema": self.schema,
                   "resolved_schema": self.schema or inspector.default_schema_name,
                   "columns": [{"name": c["name"], "sql_type": str(c["type"]),
                                "nullable": c.get("nullable"), "default": c.get("default"),
                                "computed": c.get("computed"), "identity": c.get("identity")}
                               for c in columns],
                   "unavailable": []}
            for key, method in [("primary_key", "get_pk_constraint"), ("foreign_keys", "get_foreign_keys"),
                                ("unique_constraints", "get_unique_constraints"),
                                ("indexes", "get_indexes"), ("check_constraints", "get_check_constraints")]:
                try: row[key] = getattr(inspector, method)(name, schema=self.schema)
                except NotImplementedError: row[key] = None; row["unavailable"].append(key)
            # Reject unsupported catalog values before publishing any data batch.
            _encode(row)
            result.append(row)
        return result

    def iter_table(self, table, *, batch_size=1000, columns=None, equals=None):
        """Stream driver values without NumPy inference or SQL type coercion."""
        _size(batch_size, "batch_size")
        if self._connection is None:
            with self as snapshot: yield from snapshot.iter_table(table, batch_size=batch_size, columns=columns, equals=equals)
            return
        import sqlalchemy as sa
        info = self.catalog([table])[0]
        names = [c["name"] for c in info["columns"]]
        selected = names if columns is None else list(columns)
        if not selected or len(set(selected)) != len(selected) or any(c not in names for c in selected):
            raise ValueError("select distinct existing columns")
        source = sa.table(table,*(sa.column(name) for name in names),schema=self.schema)
        # Untyped column expressions retain the DBAPI values. For example,
        # SQLite NUMERIC integers must not pass through a floating Decimal processor.
        statement = sa.select(*(sa.column(name) for name in selected)).select_from(source)
        for name,value in (equals or {}).items():
            if name not in names: raise ValueError("filter names must identify existing columns")
            statement = statement.where(source.c[name] == value)
        keys = (info["primary_key"] or {}).get("constrained_columns", [])
        if keys: statement = statement.order_by(*(source.c[key] for key in keys))
        with self._connection.execute(statement, execution_options={
                "stream_results": True, "yield_per": batch_size, "max_row_buffer": batch_size}) as result:
            while True:
                rows = result.fetchmany(batch_size)
                if not rows: return
                yield tuple(tuple(row) for row in rows)

    def read(self, table, *, columns=None, equals=None, limit=100):
        _size(limit, "limit")
        if self._connection is None:
            with self as snapshot: return snapshot.read(table,columns=columns,equals=equals,limit=limit)
        info = self.catalog([table])[0]
        names = [c["name"] for c in info["columns"]] if columns is None else list(columns)
        rows = []
        stream = self.iter_table(table, batch_size=min(limit,1000), columns=columns, equals=equals)
        try:
            for batch in stream:
                rows.extend(dict(zip(names,row,strict=True)) for row in batch[:limit-len(rows)])
                if len(rows) >= limit: break
        finally: stream.close()
        return rows

    def attach_to_hive(self, hive, prefix="sql"):
        names = []
        for operation,handler in [("catalog", lambda data, **kw: self.catalog(**data)),
                                  ("read", lambda data, **kw: self.read(**data))]:
            name = prefix+"."+operation
            hive.add_worker(name,handler,capability="analyze",worker_type="sql-source:"+operation)
            names.append(name)
        return names


def _batch(info, rows, offset, import_id):
    from rexgraph import RexGraph
    count = len(rows)
    graph = RexGraph.from_hypergraph(np.arange(count+1,dtype=np.int32),
        np.arange(count,dtype=np.int32), relation_ids=np.arange(offset,offset+count,dtype=np.int64))
    graph.set_provenance({"source": "sql-import", "sql_batch": {
        "version": 1, "import_id": import_id, "table": info["table"], "schema": info["schema"],
        "columns": [c["name"] for c in info["columns"]], "row_offset": offset,
        "row_count": count, "cell_encoding": "sql-scalar-v1",
        "ordering": "primary-key" if (info["primary_key"] or {}).get("constrained_columns") else "observed-driver-order"}})
    for i,row in enumerate(rows):
        graph.attach_metadata(1,i,"sql_row",tuple(_encode(value) for value in row))
    return graph


def import_sql(source, store, record_id, *, expected_version=0, tables=None,
               schema=None, batch_size=1000, isolation=None, progress=None):
    """Publish verified row batches, then one complete manifest with a revision guard."""
    from agent.relational import RelationalSession
    from rexgraph import RexGraph
    from rexgraph.object_identity import object_digest
    from rcdb import VersionConflictError
    _size(batch_size,"batch_size")
    session = RelationalSession(store)
    current = session.store.get_record(record_id)
    if type(expected_version) is not int or expected_version < 0:
        raise ValueError("expected_version must be a nonnegative integer")
    if (0 if current is None else current.version) != expected_version:
        raise VersionConflictError("SQL import destination changed")
    import_id = uuid.uuid4().hex
    table_records = []
    selected = source if isinstance(source,SQLSource) else SQLSource(source,schema=schema,isolation=isolation)
    with (selected if selected._connection is None else nullcontext(selected)) as selected:
        catalog = selected.catalog(tables)
        isolation_level = selected._connection.get_isolation_level()
        for index,info in enumerate(catalog):
            total, references = 0, []
            iterator = selected.iter_table(info["table"],batch_size=batch_size)
            try:
                for number,rows in enumerate(iterator):
                    graph = _batch(info,rows,total,import_id)
                    key = f"{record_id}:import:{import_id}:{index}:{number}"
                    receipt = session.publish(key,graph,expected_version=0,
                        meta={"sql_import_id":import_id,"sql_table":info["table"],"row_offset":total})
                    actual = session.read(key,receipt.version)
                    if object_digest(actual.value) != object_digest(graph):
                        raise ValueError("published SQL batch differs from its complete source")
                    references.append({"record_id":key,"version":receipt.version,
                        "state_digest":actual.state_digest,"rows":len(rows)})
                    total += len(rows)
                    if progress: progress({"table":info["table"],"rows":total,"batches":len(references)})
            finally: iterator.close()
            # Empty tables have an explicit schema-bearing native batch too.
            if not references:
                graph = _batch(info,(),0,import_id)
                key=f"{record_id}:import:{import_id}:{index}:0"
                receipt=session.publish(key,graph,expected_version=0,meta={"sql_import_id":import_id})
                snapshot=session.read(key,receipt.version)
                if object_digest(snapshot.value)!=object_digest(graph):raise ValueError("empty SQL batch changed")
                references.append({"record_id":key,"version":receipt.version,"state_digest":snapshot.state_digest,"rows":0})
            table_records.append({"declaration":info,"rows":total,"batches":references})
        declaration={"version":1,"import_id":import_id,"origin":selected.origin(),
            "observed_at":datetime.now().astimezone().isoformat(),"isolation_level":isolation_level,
            "consistency":"one read transaction; snapshot guarantees depend on the source isolation level",
            "selection":"all tables in selected schema" if tables is None else "explicit tables",
            "tables":table_records,"complete":True,"cell_encoding":"sql-scalar-v1"}
    manifest=RexGraph.from_hypergraph([0,1],[0]);manifest.attach_metadata(1,0,"sql_import",declaration)
    receipt=session.publish(record_id,manifest,expected_version=expected_version,meta={"sql_import_id":import_id})
    return {"record":receipt.to_wire_dict(),"import_id":import_id,"complete":True,
            "tables":len(table_records),"rows":sum(table["rows"] for table in table_records),
            "batches":sum(len(table["batches"]) for table in table_records)}


class SQLDataset:
    """Read a pinned complete import without its original SQL service or driver."""
    def __init__(self, store, record_id, *, version=None):
        from agent.relational import RelationalSession
        self.session=RelationalSession(store)
        self.snapshot=self.session.read(record_id,version)
        from rexgraph import RexGraph
        if not isinstance(self.snapshot.value,RexGraph):raise ValueError("selected record is not a SQL dataset")
        self.manifest=self.snapshot.value.get_metadata(1,0,"sql_import")
        if not isinstance(self.manifest,dict) or self.manifest.get("version")!=1 or self.manifest.get("complete") is not True:
            raise ValueError("selected record is not a complete SQL import")
        self._loader=self._stored

    def _stored(self, reference):
        snapshot=self.session.read(reference["record_id"],reference["version"])
        if snapshot.state_digest!=reference["state_digest"]:raise ValueError("SQL import batch identity changed")
        return snapshot.value

    def rows(self,table):
        matches=[entry for entry in self.manifest["tables"] if entry["declaration"]["table"]==table]
        if len(matches)!=1:raise KeyError(table)
        entry=matches[0];names=[column["name"] for column in entry["declaration"]["columns"]]
        total=0
        for reference in entry["batches"]:
            graph=self._loader(reference);info=graph.provenance.get("sql_batch",{})
            if (info.get("import_id")!=self.manifest["import_id"] or info.get("table")!=table
                    or info.get("row_offset")!=total or info.get("columns")!=names
                    or info.get("row_count")!=reference["rows"] or graph.nE!=reference["rows"]):
                raise ValueError("SQL import batch does not match its manifest")
            for i in range(graph.nE):
                values=graph.get_metadata(1,i,"sql_row")
                yield dict(zip(names,(_decode(value) for value in values),strict=True))
            total+=graph.nE
        if total!=entry["rows"]:raise ValueError("SQL import row count differs from its manifest")

    def export(self,directory,*,format="rcbd"):
        from rexgraph.io import save,load
        from rexgraph.io.publication import publication_lock,staged_publication
        from rexgraph.object_identity import object_digest
        suffix={"rcbf":".rcbf","rcbd":".rcbd","safetensors":".safetensors","hdf5":".h5"}.get(format)
        if suffix is None:raise ValueError("SQL dataset export supports rcbf, rcbd, safetensors or hdf5")
        target=Path(directory);target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():raise FileExistsError(target)
        with ExitStack() as locks, staged_publication(target,directory=True) as root:
            manifest_name="dataset"+suffix;save(root/manifest_name,self.snapshot.value,format=format)
            index={"version":1,"format":format,"manifest":manifest_name,
                "manifest_digest":object_digest(self.snapshot.value),"batches":[]}
            for entry in self.manifest["tables"]:
                for reference in entry["batches"]:
                    graph=self._loader(reference);name=f"batch-{len(index['batches']):08d}"+suffix
                    save(root/name,graph,format=format)
                    if object_digest(load(root/name,format=format))!=object_digest(graph):
                        raise ValueError("SQL export file differs from its native batch")
                    index["batches"].append({**reference,"file":name,"object_digest":object_digest(graph)})
            if object_digest(load(root/manifest_name,format=format))!=index["manifest_digest"]:
                raise ValueError("SQL export manifest differs from its native source")
            (root/"EXPORT.json").write_text(json.dumps(index,indent=2),encoding="utf-8")
            locks.enter_context(publication_lock(target.parent))
            if target.exists():raise FileExistsError(target)
        return str(target)

    @classmethod
    def from_directory(cls,directory):
        from rexgraph.io import load
        from rexgraph.object_identity import object_digest
        root=Path(directory).resolve();index=json.loads((root/"EXPORT.json").read_text())
        if index.get("version")!=1 or index.get("format") not in {"rcbf","rcbd","safetensors","hdf5"}:
            raise ValueError("unsupported SQL dataset export")
        def selected(name):
            path=(root/name).resolve()
            if path==root or root not in path.parents:raise ValueError("SQL export file escapes its directory")
            return load(path,format=index["format"])
        manifest=selected(index["manifest"])
        if object_digest(manifest)!=index["manifest_digest"]:raise ValueError("SQL export manifest identity changed")
        from types import SimpleNamespace
        result=cls.__new__(cls);result.snapshot=SimpleNamespace(value=manifest)
        result.manifest=manifest.get_metadata(1,0,"sql_import")
        if not isinstance(result.manifest,dict) or result.manifest.get("complete") is not True:
            raise ValueError("SQL export has no complete manifest")
        files={(r["record_id"],r["version"]):r for r in index["batches"]}
        if len(files)!=len(index["batches"]):raise ValueError("duplicate SQL export batch address")
        def loader(reference):
            record=files[(reference["record_id"],reference["version"])];graph=selected(record["file"])
            if (record["state_digest"]!=reference["state_digest"] or object_digest(graph)!=record["object_digest"]
                    or object_digest(graph)!=reference["state_digest"]):
                raise ValueError("SQL export batch identity changed")
            return graph
        result._loader=loader
        return result


def schema_model(catalog):
    """Declare the selected schema view; external-schema references stay in its catalog."""
    from agent.schema_complex import SchemaModel, TableDef, ForeignKey
    tables=[];foreign=[]
    for info in catalog:
        names=[column["name"] for column in info["columns"]]
        pk=list((info["primary_key"] or {}).get("constrained_columns", []))
        tables.append(TableDef(name=info["table"],columns=names,primary_key=pk))
        nullable={column["name"]:column["nullable"] for column in info["columns"]}
        for fk in info["foreign_keys"] or ():
            if fk.get("referred_schema") not in (None,info.get("resolved_schema",info["schema"])):
                continue
            columns=list(fk["constrained_columns"])
            foreign.append(ForeignKey(from_table=info["table"],from_cols=columns,
                to_table=fk["referred_table"],to_cols=list(fk["referred_columns"]),
                nullable=any(nullable.get(c) is not False for c in columns),
                identifying=bool(columns) and set(columns)<=set(pk),
                on_delete=fk.get("options",{}).get("ondelete", "")))
    return SchemaModel(tables=tables,foreign_keys=foreign)
