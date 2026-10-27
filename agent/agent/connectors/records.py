"""Owning Core complexes read from selected RCDB revisions or native files."""
from . import BaseConnector, Capabilities, ConnectorError


def _complex(value, source):
    from rexgraph import RexGraph
    if not isinstance(value, RexGraph):
        raise ConnectorError("this reader requires a Core complex; use typed RCQL for other native values")
    labels = value.vertex_labels
    return value, {"source": source, "vertex_labels": list(labels) if labels is not None else [],
                   "nV": value.nV, "nE": value.nE}


class RCDBConnector(BaseConnector):
    CAPABILITIES = Capabilities(schemes=("rcdb",))

    def select(self, source):
        from agent.relational import RelationalSession
        if not isinstance(source, dict) or set(source) - {"record_id", "version", "store"}:
            raise ConnectorError("RCDB sources require record_id and optional explicit version/store")
        session = RelationalSession(source.get("store"))
        selected = session.read(source["record_id"], source.get("version"))
        return session, selected

    def read(self, source):
        _, selected = self.select(source)
        value, meta = _complex(selected.value, "rcdb:" + selected.record.id)
        return value, {**meta, "record_id": selected.record.id, "version": selected.record.version,
                       "state_digest": selected.state_digest}


class NativeFileConnector(BaseConnector):
    CAPABILITIES = Capabilities(schemes=("rcbd", "rcbf"))

    def read(self, source):
        from agent.native_files import load_native_file as load
        if isinstance(source, dict):
            if set(source) != {"path", "format"}:
                raise ConnectorError("native file selection requires path and format")
            value = load(source["path"], format=source["format"])
        else:
            if isinstance(source, str) and source.startswith(("rcbd://", "rcbf://")):
                format, path = source.split("://", 1)
                value = load(path, format=format)
            else:
                value = load(source)
        return _complex(value, "native-file")
