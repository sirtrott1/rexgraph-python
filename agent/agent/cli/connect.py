"""
agent.cli.connect: the connector command line (``rexgraph-connect``).

A thin CLI over :mod:`agent.connectors.service`; it adds no logic the HTTP route
doesn't also get. Point it at any source and it lists, validates, reads, or
ingests: the same four verbs the app exposes.

    rexgraph-connect list

RCDB record selections and custom file readers share the same source registry.
"""

from __future__ import annotations

import argparse
import json as _json
import sys

from agent.connectors import service as svc


def _resolve(source: str) -> str:
    """A saved connection name -> its URI (via the SecretStore); otherwise the
    source is already a URI/scheme and is returned unchanged."""
    if "://" in source:
        return source
    try:
        from agent.secrets import open_secret_store
        return open_secret_store().get(source)
    except Exception:
        return source            # treat as a bare scheme (e.g. "ontology")


def main(argv: list | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="rexgraph-connect",
        description="Connect any source to the engine: list / read / validate / ingest.")
    sub = p.add_subparsers(dest="command")

    pl = sub.add_parser("list", help="List connectors, capabilities, driver status")
    pl.add_argument("--json", action="store_true")

    pr = sub.add_parser("read", help="Build the complex read-only and summarize")
    pr.add_argument("source", help="connection URI or saved-connection name")
    pr.add_argument("--weights", action="store_true", help="request declared source weights")
    pr.add_argument("--json", action="store_true")

    pv = sub.add_parser("validate", help="Run the validation harness (exit 0/1)")
    pv.add_argument("source", help="connection URI or saved-connection name")
    pv.add_argument("--json", action="store_true")

    pi = sub.add_parser("ingest", help="Build + persist structure into an RCStore")
    pi.add_argument("source", help="connection URI or saved-connection name")
    pi.add_argument("--store", required=True, help="RCDB provider URI (local://… or memory://)")
    pi.add_argument("--id", required=True, dest="record_id", help="record id to store as")
    pi.add_argument("--tags", default="", help="comma-separated tags")
    pi.add_argument("--weights", action="store_true")
    pi.add_argument("--json", action="store_true")

    ps = sub.add_parser("import-sql", help="Import complete SQL tables into RCDB and optional native files")
    ps.add_argument("source", nargs="?", help="SQL URI or saved connection name")
    ps.add_argument("--source-env", help="environment variable containing the SQL URI")
    ps.add_argument("--store", required=True, help="RCDB destination, e.g. local:///data/records")
    ps.add_argument("--id", default="sql-import", dest="record_id")
    ps.add_argument("--expected-version", type=int, default=0)
    ps.add_argument("--schema")
    ps.add_argument("--table", action="append", dest="tables")
    ps.add_argument("--batch-size", type=int, default=1000)
    ps.add_argument("--files", help="export verified native files to a new directory")
    ps.add_argument("--format", choices=["rcbf","rcbd","safetensors","hdf5"], default="rcbd")
    ps.add_argument("--json", action="store_true")

    args = p.parse_args(argv)

    if args.command == "import-sql":
        import os
        from contextlib import closing
        from agent.rcdb import open_store
        from agent.sql_source import import_sql, SQLDataset
        if bool(args.source) == bool(args.source_env):
            p.error("select a SQL source or --source-env")
        uri=os.environ.get(args.source_env,"") if args.source_env else _resolve(args.source)
        if not uri:p.error("SQL source environment variable is empty")
        with closing(open_store(args.store)) as store:
            result=import_sql(uri,store,args.record_id,expected_version=args.expected_version,
                schema=args.schema,tables=args.tables,batch_size=args.batch_size)
            if args.files:
                result["files"]=SQLDataset(store,args.record_id,version=result["record"]["version"]).export(args.files,format=args.format)
        print(_json.dumps(result,indent=2) if args.json else
              f"Imported {result['rows']} rows from {result['tables']} tables into {args.store}; "
              f"record {args.record_id} version {result['record']['version']} (verified)")
        return 0

    if args.command == "list":
        rows = svc.list_connectors()
        if args.json:
            print(_json.dumps(rows, indent=2))
            return 0
        for c in rows:
            schemes = ", ".join(
                s["scheme"] + ("" if s["driver_available"] else " (driver missing)")
                for s in c["schemes"])
            print(f"{c['connector']:20s} [{c['capabilities']}]")
            print(f"    schemes: {schemes}")
        return 0

    if args.command == "read":
        uri = _resolve(args.source)
        out = svc.read(uri, **svc.weight_kwargs(uri, args.weights))
        print(_json.dumps(out, indent=2) if args.json else _fmt_summary(out))
        return 0

    if args.command == "validate":
        uri = _resolve(args.source)
        report = svc.validate(uri)
        if args.json:
            print(_json.dumps({"connector": report.connector, "ok": report.ok,
                               "checks": [{"name": c.name, "passed": c.passed,
                                           "detail": c.detail} for c in report.checks]},
                              indent=2))
        else:
            print(report)
        return 0 if report.ok else 1

    if args.command == "ingest":
        uri = _resolve(args.source)
        tags = [t.strip() for t in args.tags.split(",") if t.strip()]
        out = svc.ingest(uri, args.record_id, store_uri=args.store, tags=tags,
                         **svc.weight_kwargs(uri, args.weights))
        print(_json.dumps(out, indent=2) if args.json else
              f"stored '{out['stored_as']}' in {out['store']} "
              f"(nV={out['nV']} nE={out['nE']} betti={out['betti']})")
        return 0

    p.print_help()
    return 0


def _fmt_summary(o: dict) -> str:
    return (f"source: {o['source']}\n"
            f"  nV={o['nV']} nE={o['nE']} nF={o['nF']} "
            f"betti={o['betti']} chain_valid={o['chain_valid']}\n"
            f"  weighted={o['weighted']} modality={o['modality']}")


if __name__ == "__main__":
    sys.exit(main())
