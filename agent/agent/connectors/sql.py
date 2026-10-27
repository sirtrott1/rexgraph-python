"""Optional read-only SQL schema compatibility through the common SQL source."""
from . import BaseConnector, Capabilities, ConnectorError as ConnectorError
from agent.record_access import SQL_SCHEMES


class SQLConnector(BaseConnector):
    def __init__(self, with_weights=False):
        self.with_weights = with_weights

    def capabilities(self):
        return Capabilities(modality=True, faces=True, weights=self.with_weights,
                            schemes=tuple(sorted(SQL_SCHEMES)))

    def read(self, source):
        from agent.sql_source import SQLSource, schema_model
        from agent.schema_complex import schema_to_rex
        from rexgraph import RexGraph
        with (source if isinstance(source,SQLSource) else SQLSource(source)) as selected:
            catalog = selected.catalog()
            model = schema_model(catalog)
            counts = None
            if self.with_weights:
                from agent.schema_complex import _row_count
                counts = {info["table"]:_row_count(selected._connection,selected.engine,info["table"])
                          for info in catalog}
            weights = None if counts is None else {
                f"{fk.from_table}->{fk.to_table}":counts[fk.from_table] for fk in model.foreign_keys}
            graph, metadata = schema_to_rex(model,weights=weights)
            if graph is None:
                graph = RexGraph.from_cells([len(catalog)])
                metadata = {"vertex_labels": [table["table"] for table in catalog], "edges": []}
            metadata = {**metadata, "source": "sql-source", "sql_catalog": catalog,
                        "sql_origin": selected.origin()}
            metadata["modality"] = [{"from_columns": list(fk.from_cols),
                "to_columns": list(fk.to_cols), "nullable": fk.nullable,
                "identifying": fk.identifying, "on_delete": fk.on_delete}
                for fk in model.foreign_keys if (fk.from_table,fk.to_table) in
                    {tuple(edge) for edge in metadata.get("edges", [])}]
            if counts is not None:
                metadata["weights"] = [counts[a] for a,b in metadata.get("edges", [])]
            graph.set_provenance(metadata)
            return graph, metadata
