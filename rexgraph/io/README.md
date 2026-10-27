# RexGraph I/O

A table tells the reader where its columns are; it does not tell the application
which columns define a relation. The I/O package keeps those choices separate.
A schema controls field types and number conversion, and a relation mapping
selects participants, weights and other declarations.

Saving the resulting complex is a different choice. A native container carries
its complete registered state, while a table export carries the columns in its
schema. Use the native form when the next application must reconstruct the same
complex, including exact values and presence information.

## Installation

`make install` includes the file format dependencies. `make install-core`
provides the native RCBD container with the base library. For an existing Core
environment, run `python -m pip install ".[io]"` from the repository root to add
the optional format readers and writers. Encryption and signatures use the
`security` extra. See the [installation guide](../../README.md#installation).

## Saving a complex

```python
from rexgraph import RexGraph
from rexgraph.io import load_rcbd, save_rcbd

rex = RexGraph.from_graph([0, 1, 2], [1, 2, 0])
save_rcbd("cycle.rcbd", rex)
restored = load_rcbd("cycle.rcbd")
assert tuple(restored.betti) == (1, 1, 0)
```

Native state retains exact coefficient kinds, presence, identities, declared
heads and shares, and registered components. Container readers check the saved
state before reconstructing the object. HDF5 and Zarr also provide native graph
storage; their container content seals are separate from the graph's semantic
state identity.

| Format | Use | Dependency |
| --- | --- | --- |
| RCBD | Native complex and temporal bundles | Core |
| RCBO | Complete ontology snapshot, selected anchor and source identity | Core |
| RCBF | Complete canonical state in a sequential binary file; legacy snapshot import | Core |
| Native JSON | Complete graph or temporal state in a versioned JSON envelope | Core |
| safetensors | Native tensor state and model transport | safetensors |
| HDF5 | Native graphs, arrays, caches and result groups | h5py |
| Zarr | Native graphs, arrays, caches and result groups | Zarr |
| Arrow IPC | Typed columnar records and native state transport | pyarrow |
| Parquet | Declared tables and exact coefficient columns | pyarrow |
| SQL | Declared boundary tables and checked native state | SQLAlchemy |
| CSV, JSON, JSONL | Source records with an explicit schema | Core |
| XLSX | Declared workbook records | openpyxl |

A table export contains the columns selected by its schema. Use a native
container to preserve a complete complex. Arrow and Parquet coefficient
columns preserve integer and rational kinds, presence and large integers;
an undeclared float in an exact column requires an explicit conversion rule.

`save_rcbf(path, graph)` writes complete canonical static or temporal state in
an RCBF version 6 sequential file. `load_rcbf(path)` checks its framing, checksum
and canonical state. RCBD uses addressable directory members for the same state.
Both can be loaded, put into RCDB and queried through RCQL. Older RCBF readers
that accept only versions 4 and 5 cannot read version 6. The version 6 byte budget
defaults to 1 GiB; pass `max_bytes` to set an explicit different limit.

Legacy RCBF versions 4 and 5 retain their snapshot import rules. Stored temporal
states require `allow_current_snapshot=True` to request only the current snapshot;
that option does not import their history. Mixed directedness, inexact C2
coefficients, duplicate signal names and unclaimed timeless tails are rejected
by that legacy importer.

The C++ RFC formats use their own owning readers and admission limits. Generic
`load` does not interpret `.rfcx`, `.rfcxs`, `.rfcxa` or `.rfcxg` as a Core
container. For a supported algebraic RFCXA archive, retain its complete bytes
through the bridge's record adapter:

```python
from pathlib import Path
from rexgraph_cpp import NativeAlgebraicArchive
from rexgraph.io.algebraic_record import algebraic_record, read_algebraic_record

archive = NativeAlgebraicArchive(Path("fields.rfcxa").read_bytes())
save_rcbd("fields.rcbd", algebraic_record(archive))
restored = read_algebraic_record(load_rcbd("fields.rcbd"))
assert restored.archive == archive.archive
```

RFCXG graphs and RFCXS histories have distinct adapters and retain their named
owners and source cutoffs. See the [bridge usage guide](../../bindings/cpp/README.md).
An extension alone does not declare a transport between these carriers.

`save("graph.json", rex)` writes complete canonical state, including exact
values, relation identities and registered fields. Its `rexgraph-state-v1`
envelope contains a base64 native value payload; `load` validates the restored
state. Earlier graph JSON and external JSON source formats remain readable.
`rex.to_json()` remains the older graph view export and does not carry the full
state. JSON state transport does not calculate a layout or spectral diagnostics.

## Reading a weighted table

Save this input as `edges.csv`:

```csv
source,target,weight
A,B,1.0
B,C,0.5
C,A,0.25
```

Declare the participant labels as strings and the weights as exact decimals:

```python
from fractions import Fraction
from rexgraph import NumberRule, RelationSpec, RexGraph, ValueRules
from rexgraph.construction import construct
from rexgraph.io.readers import read_batches
from rexgraph.io.records import RecordField, RecordSchema

schema = RecordSchema((
    RecordField("source", "string"),
    RecordField("target", "string"),
    RecordField("weight", "number", rules=ValueRules(NumberRule.DECIMAL_EXACT)),
))
records = read_batches("edges.csv", reader="csv", schema=schema)
relations = construct(records, RelationSpec("pair", ("source", "target"), weight="weight"))
rex = RexGraph.from_relations(relations)
assert relations.weight.values()[1] == Fraction(1, 2)
```

`read_batches` supports CSV, JSON, JSONL, Arrow IPC, Parquet, read only SQLite
and XLSX. The schema declares field types, source number rules and supported
limits; the relation mapping declares how records become cells. An explicit
vertex domain can retain entities absent from the relation table.

## Reusing declarations and processing batches

`DatasetDeclaration` groups the reader, schema, relation mapping, source options
and optional vertex domain. Applications can register it with RCQL or System
to expose a named dataset. Query text resolves the registered name; the
application supplies the source file or stream.

Configure batch sizes and record byte limits through the reader's declared
options. Streaming bounds the input batches; the final relational complex still
occupies memory proportional to its retained cells and coefficients. The
[reader reference](REFERENCE.md#declared-reader-capabilities) lists source and
streaming contracts.

Arrow IPC streams can be forward only. JSON arrays use bounded whole document
storage and accept fragmented reads; JSONL and CSV yield batches as records
arrive. Caller supplied streams remain open when iteration ends or is cancelled.
For indexed RCBD access, use logical tensor names such as
`bundle["checkpoint/0/boundary_idx"]`; the bundle handles filename encoding.

Arrow array streaming slices oversized file batches to the requested positive
`batch_rows` limit. Matrix rows that cross file batches are buffered until
complete; differently shaped arrays retain their own row boundaries. IPC reads
use a memory map, and yielded NumPy arrays retain their data after the stream is
closed. Arrow IPC and Parquet writers stage output before publication so a
failed write preserves an existing file.

Agent's `auto_rex` offers automatic adapters for common files, arrays, DataFrames
and text. Use the declared readers when source typing, relation meaning and
number conversion must be controlled by the application. Compatibility
`csv_loader` and `json_loader` APIs retain their own inference and conversion
behavior.

## Integrity, security and compatibility

Native state codecs check dimensions, component ownership, coefficients and
tensor declarations. HDF5 and Zarr writers stage complete output before
publishing it, preserving the preceding output if encoding fails. SQL boundary
publication stores native state and its projection in one transaction and
checks their agreement on reconstruction.

Content digests detect corruption and establish identity. Encryption and
authenticated signatures require configured keys. Older native formats have
explicit compatibility readers; consuming an unsealed legacy cache requires
the corresponding compatibility option. See the
[container contracts](REFERENCE.md#current-native-contracts) and
[format API reference](REFERENCE.md) for publication, security and migration
details.

RCDB provides [versioned records](../../rcdb/README.md), and RCQL provides
[portable query results](../../rcql/README.md#portable-exact-results). Both use
the shared Core state codecs.

## Ontology files

`.rcbo` is the native container for an `OntologyState`. `save_ontology` and
`load_ontology` retain the complete source, selected anchor, exact declarations
and ontology identity. The current payload uses the complete RCBF state codec
and needs no optional tensor backend. Declared legacy safetensors payloads
require that backend when read and retain their original bytes.

`write_rcbo(stream, ontology)` and `read_rcbo(stream)` support binary streams
without seeking or closing the caller's stream. Framing includes the version,
source codec, bounded lengths, identities and a checksum. The default whole-file
limit is 256 MiB; `max_bytes` can lower it. Truncation, trailing data, checksum
errors and unsupported versions fail before graph reconstruction.
`save_rcbo` publishes atomically, preserving an existing file if writing fails.

`save_ontology` also accepts ordinary complete graph containers such as RCBF,
RCBD and safetensors. Use `load_ontology(..., anchor=(grade, index))` when the
selected ontology is in an ordinary graph container. An RCBO file retains its
own anchor. FileCatalog and Agent readers return its complete graph so existing
query and analysis paths retain the ontology declaration. See the
[ontology example](../README.md#ontology-snapshots).
