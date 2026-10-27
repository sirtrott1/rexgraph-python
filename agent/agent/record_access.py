"""Agent store admission and typed reads over RCDB records."""
from urllib.parse import urlparse

_STORE_SCHEMES = {"", "auto", "local", "memory", "file", "rex",
                  "s3", "gs", "gcs", "az", "abfs", "adl"}
SQL_SCHEMES = frozenset({"sqlite", "duckdb", "postgres", "postgresql", "mysql",
    "mariadb", "mssql", "sqlserver", "oracle", "snowflake", "redshift",
    "bigquery", "databricks", "trino", "presto", "clickhouse", "access"})


def register_store_scheme(scheme):
    if not isinstance(scheme, str) or not scheme or scheme.split("+", 1)[0].lower() in SQL_SCHEMES:
        raise ValueError("Agent store schemes must name an RCDB provider")
    _STORE_SCHEMES.add(scheme.lower())


def check_store_uri(uri):
    """Admit RCDB file providers without opening a database connection."""
    if not isinstance(uri, str) or not uri:
        raise ValueError("a nonempty RCDB store URI is required")
    scheme = urlparse(uri).scheme.lower()
    if scheme.split("+", 1)[0] in SQL_SCHEMES or scheme not in _STORE_SCHEMES:
        raise ValueError("Agent record stores require RCDB providers; use SQLSource or import-sql for SQL data")
    return uri


def require_store(store):
    """Reject SQL-backed stores, including existing workspace wrappers."""
    seen = set()
    current = store
    while id(current) not in seen:
        seen.add(id(current))
        if getattr(current, "backend", None) == "sql":
            raise ValueError("Agent requires an RCDB file or memory provider, not a SQL backend")
        try:
            inner = vars(current).get("_inner", current)
        except TypeError:
            inner = current
        if inner is current:
            break
        current = inner
    return store


def read_query(store, text, *, parameters=None):
    """Execute a read-only RCQL query against an explicitly admitted store."""
    from rcql import BoundSource, Executor, MutationQuery, SourcePolicy, parse
    require_store(store)
    query = parse(text) if isinstance(text, str) else text
    if isinstance(query, MutationQuery):
        raise PermissionError("this operation is read-only")
    policy = SourcePolicy.allow("read", "records", "identity", "history", "search")
    return Executor(sources={"db": BoundSource(store, policy)}, params=parameters).execute(query)


def record_pin_key(pin):
    """Validate a selected record reference, including its actual store identity."""
    if not isinstance(pin, dict) or set(pin) != {"store_id", "record_id", "version", "state_digest"}:
        raise ValueError("record references require store_id, record_id, version and state_digest")
    if any(not isinstance(pin[k], str) or not pin[k] for k in ("store_id", "record_id", "state_digest")):
        raise ValueError("record reference identities must be nonempty strings")
    if type(pin["version"]) is not int or pin["version"] < 1:
        raise ValueError("record reference version must be positive")
    return tuple(pin[k] for k in ("store_id", "record_id", "version", "state_digest"))


def copy_record_pin(pin):
    """Detach caller-owned reference dictionaries before I/O or native execution."""
    if not isinstance(pin, dict):
        record_pin_key(pin)
    snapshot = dict(pin)
    record_pin_key(snapshot)
    return snapshot


class PinnedRecords:
    """Selected snapshots and current publication checks under an existing session.

    Historical selection verifies its exact version and payload. Current checks
    inspect published envelopes, so guards need not decode whole stored fields.
    Workspace and actor admission remain the session store's responsibility.
    """

    def __init__(self, session):
        self.session = session

    def pin(self, record_id, version=None):
        snapshot = self.session.read(record_id, version)
        return dict(store_id=self.session.store.store_id, record_id=snapshot.record.id,
                    version=snapshot.record.version, state_digest=snapshot.state_digest)

    def current(self, pin):
        from rcdb.engine import metadata_digest
        pin = copy_record_pin(pin)
        if pin["store_id"] != self.session.store.store_id:
            raise ValueError("record reference belongs to another store")
        record = self.session.store.get_record(pin["record_id"])
        if record is None or (record.id, record.version) != (pin["record_id"], pin["version"]):
            raise ValueError("selected record is no longer current")
        envelope = record.envelope
        if envelope is None:
            if self.pin(pin["record_id"]) != pin:
                raise ValueError("selected record digest differs")
            return
        envelope.check_address(store_id=pin["store_id"], record_id=pin["record_id"], record_version=pin["version"])
        if envelope.object_digest != pin["state_digest"] or envelope.metadata_digest != metadata_digest(record):
            raise ValueError("selected record envelope differs")

    def read(self, pin, *, current=False):
        pin = copy_record_pin(pin)
        if pin["store_id"] != self.session.store.store_id:
            raise ValueError("record reference belongs to another store")
        if current:
            self.current(pin)
        snapshot = self.session.read(pin["record_id"], pin["version"])
        if (snapshot.record.id, snapshot.record.version, snapshot.state_digest) != (
                pin["record_id"], pin["version"], pin["state_digest"]):
            raise ValueError("selected record version or digest differs")
        return snapshot.value

    def value(self, pin, alias, *, current=False):
        from rcql import Result
        value = self.read(pin, current=current)
        if not isinstance(value, Result) or value.aliases != (alias,) or len(value.values) != 1:
            raise ValueError(f"selected record must contain one {alias} Result value")
        return value.values[0]
