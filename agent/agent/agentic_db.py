"""RCDB records, RCQL programs and custom files exposed as Hive capabilities."""
from __future__ import annotations

from .record_access import check_store_uri, require_store
from .relational import RelationalSession


class AgenticDB:
    """An admitted RCDB store under one existing Agent caller context.

    Opening a URI is a local application operation. Socket tools use the bound
    session and uploaded file handles rather than accepting store paths.
    """

    def __init__(self, source=None, *, store=None, writable=False, context=None):
        from . import rcdb
        if type(writable) is not bool:
            raise TypeError("writable requires a boolean")
        if source is not None and store is not None:
            raise ValueError("select one RCDB source or store")
        self._owned = isinstance(source, str)
        selected = rcdb.open_store(check_store_uri(source)) if self._owned else source
        selected = selected if selected is not None else store
        selected = rcdb.default_store() if selected is None else require_store(selected)
        try:
            self.session = RelationalSession(selected, context=context)
        except BaseException:
            if self._owned:
                selected.close()
            raise
        self.store = self.session.store
        self.writable = writable

    def close(self):
        if self._owned:
            self.store.close()
            self._owned = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def query(self, query, *, parameters=None, sources=None):
        """Execute RCQL under the store's current policy and revision bindings."""
        return self.session.query(query, parameters=parameters, sources=sources,
                                  allow_write=self.writable)

    def records(self, *, filters=None, limit=100, offset=0):
        return self.query('FROM $db RETURN RCDB_SELECT($filters,$limit,$offset)',
            parameters={"filters": filters, "limit": limit, "offset": offset}).values[0]

    def health(self):
        return self.query('FROM $db RETURN RCDB_STATS(), RCDB_SECURITY()')

    def search(self, text, *, limit=100):
        """Search declared record vocabulary without interpreting executable text."""
        return self.query('FROM $db RETURN RCDB_SEARCH($text,$limit)',
                          parameters={"text": text, "limit": limit})

    def inspect(self, record_id, *, version=None):
        return self.session.read(record_id, version)

    def extract(self, record_id, *, version=None):
        """Return the selected complete value, retaining its native owner."""
        return self.inspect(record_id, version=version).value

    def publish(self, record_id, value, *, expected_version, meta=None):
        if not self.writable:
            raise PermissionError("this AgenticDB is read-only")
        return self.session.publish(record_id, value, expected_version=expected_version, meta=meta)

    def modify(self, query, *, parameters=None):
        """Publish only an explicit RCQL mutation with its expected version."""
        from rcql import MutationQuery, parse
        if not self.writable:
            raise PermissionError("this AgenticDB is read-only")
        query = parse(query) if isinstance(query, str) else query
        if not isinstance(query, MutationQuery):
            raise TypeError("modify requires a typed RCQL mutation")
        return self.query(query, parameters=parameters)

    def import_file(self, path, record_id, *, expected_version, format=None):
        """Publish a locally selected value using Core's installed format readers."""
        from agent.native_files import load_native_file as load
        if not self.writable:
            raise PermissionError("this AgenticDB is read-only")
        value = load(path, format=format)
        return self.publish(record_id, value, expected_version=expected_version)

    @classmethod
    def from_sql(cls, source, **options):
        """Open an optional live SQL read source while RCDB remains the Agent store."""
        from .sql_source import SQLSource
        return SQLSource(source, **options)

    def import_sql(self, source, record_id, *, expected_version=0, **options):
        """Convert complete selected SQL tables into verified native RCDB records."""
        if not self.writable:
            raise PermissionError("this AgenticDB is read-only")
        from .sql_source import import_sql
        return import_sql(source, self.store, record_id, expected_version=expected_version, **options)

    def sql_dataset(self, record_id, *, version=None):
        from .sql_source import SQLDataset
        return SQLDataset(self.store, record_id, version=version)

    def export_file(self, record_id, path, *, version=None, format=None):
        """Write the complete selected value through Core's installed format writer."""
        from rexgraph.io import save
        value = self.extract(record_id, version=version)
        save(path, value, format=format)
        return path

    def data_complex(self, record_id, *, link_on, id_col=None, version=None):
        """Construct an explicitly linked row view of one complete stored value."""
        from rcql import Result
        from .data_complex import analyze_rows
        value = self.extract(record_id, version=version)
        if isinstance(value, Result):
            if len(value.values) != 1:
                raise ValueError("row construction requires one declared result output")
            value = value.values[0]
        if not isinstance(value, (list, tuple)) or any(not isinstance(row, dict) for row in value):
            raise TypeError("row construction requires a stored sequence of records")
        result = analyze_rows(value, link_on=link_on, id_col=id_col)
        result["source"] = record_id
        return result

    def attach_to_hive(self, hive, prefix="db"):
        """Register actual RCQL and record operations as capability workers."""
        def arguments(data):
            if not isinstance(data, dict):
                raise TypeError("record worker arguments require a mapping")
            return data

        specs = [
            ("query", lambda data, **kw: self.query(**arguments(data)), "analyze"),
            ("records", lambda data, **kw: self.records(**arguments(data)), "analyze"),
            ("search", lambda data, **kw: self.search(data) if isinstance(data, str)
             else self.search(**arguments(data)), "analyze"),
            ("health", lambda data, **kw: self.health(), "analyze"),
            ("extract", lambda data, **kw: self.extract(**arguments(data)), "transform"),
        ]
        if self.writable:
            specs.extend([
                ("modify", lambda data, **kw: self.modify(**arguments(data)), "transform"),
                ("publish", lambda data, **kw: self.publish(**arguments(data)), "transform"),
            ])
        names = []
        for operation, handler, capability in specs:
            name = prefix + "." + operation
            hive.add_worker(name, handler, capability=capability,
                            worker_type="rcdb:" + operation,
                            specialties=["rcdb", "rcql", operation, prefix])
            names.append(name)
        return names
