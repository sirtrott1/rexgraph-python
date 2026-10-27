"""RCQL queries and metadata observations over records in an RCDB provider."""
from __future__ import annotations

from .record_access import read_query, require_store

COLUMNS = (
    ("id", "text"), ("version", "integer"), ("nV", "integer"),
    ("nE", "integer"), ("nF", "integer"), ("betti0", "integer"),
    ("betti1", "integer"), ("kappa_mean", "number"), ("n_voids", "integer"),
    ("chain_valid", "boolean"), ("n_labels", "integer"), ("source", "text"),
    ("object_type", "text"), ("tx_from", "number"), ("valid_from", "number"),
)


def _row(record):
    signature = record["signature"]
    betti = signature.get("betti", ())
    values = {name: signature.get(name) for name, _ in COLUMNS}
    values.update({key: record[key] for key in ("id", "version", "tx_from", "valid_from")})
    values["betti0"] = betti[0] if betti else None
    values["betti1"] = betti[1] if len(betti) > 1 else signature.get("betti1")
    return values


class SignatureView:
    """A live read-only RCQL view with explicit record selection and time cuts."""

    def __init__(self, store, *, limit=10000, include_history=False, as_of=None, valid_at=None):
        self.store = require_store(store)
        self.limit = limit
        self.include_history = include_history
        self.as_of, self.valid_at = as_of, valid_at

    def query(self, query, *, parameters=None):
        """Return a complete canonical RCQL result from the selected store."""
        return read_query(self.store, query, parameters=parameters)

    def records(self, filters=None, *, limit=None, offset=0):
        return self.query('FROM $db RETURN RCDB_SELECT($filters,$limit,$offset,$as_of,$valid_at,$history)',
            parameters={"filters": filters, "limit": self.limit if limit is None else limit,
                        "offset": offset, "as_of": self.as_of, "valid_at": self.valid_at,
                        "history": self.include_history}).values[0]

    def rows(self, filters=None, *, limit=None, offset=0):
        return [_row(record) for record in self.records(filters, limit=limit, offset=offset)]

    def ids(self, filters=None, *, limit=None, offset=0):
        """Select identities with the store's declared structural predicates."""
        return [record["id"] for record in self.records(filters, limit=limit, offset=offset)]

    def complexes(self, filters=None, *, limit=100):
        return [(record["id"], self.store.read_record(record["id"], version=record["version"]).value)
                for record in self.records(filters, limit=limit)]

    def columns(self):
        return [name for name, _ in COLUMNS]

    def refresh(self):
        """Return the live view; no independent materialized table is retained."""
        return self

    def describe(self):
        """Summarize the selected metadata without replacing stored values."""
        from fractions import Fraction
        from math import fsum
        rows = self.rows()
        result = []
        for name, kind in COLUMNS:
            values = [row[name] for row in rows if row[name] is not None]
            numeric = values if kind in {"integer", "number"} else []
            mean = (Fraction(sum(numeric), len(numeric)) if numeric and all(type(v) is int for v in numeric)
                    else fsum(numeric)/len(numeric) if numeric else None)
            result.append({"name": name, "kind": kind, "count": len(values),
                           "missing": len(rows)-len(values), "mean": mean,
                           "minimum": min(numeric) if numeric else None,
                           "maximum": max(numeric) if numeric else None})
        return result

    def to_arrow(self):
        import pyarrow
        rows = self.rows()
        return pyarrow.Table.from_pylist(rows) if rows else pyarrow.table({name: [] for name, _ in COLUMNS})

    def sql(self, query):
        raise ValueError("SQL execution was removed; use query() with RCQL")

    def close(self):
        """The caller retains ownership of the supplied RCDB store."""

    def __len__(self):
        return len(self.records())


def signature_rows(store, **options):
    return [tuple(row[name] for name, _ in COLUMNS) for row in SignatureView(store, **options).rows()]


def signature_view(store, **options):
    return SignatureView(store, **options)


def available():
    return True
