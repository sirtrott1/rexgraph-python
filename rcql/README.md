# RCQL

RCQL lets an application ask questions of a complex without discarding the
cell identities and bases needed to interpret the answer. A query can select
relations, measure homology or apply a typed operator through RexGraph. Results
carry their source references, arithmetic declarations and execution provenance
so they can be inspected or saved beyond the current query.

The application registers the available sources. Query text resolves those
names, and explicit mutation expressions publish changes through RCDB. Selecting
rows and changing the stored complex are separate operations.

## Installation

From the repository root, `make install` installs RCQL with the application
packages. Use `make python` to open the installed interpreter. If Core is
already installed, run `python -m pip install ./rcql` from the repository root.
RCQL requires Core and NumPy; RCDB sources are supplied by the application.
See the [installation guide](../README.md#installation).

## Running a query

Register a complex under a name and refer to it in the query:

```python
from rexgraph import RexGraph
from rcql import run

network = RexGraph.from_graph([0, 1, 2], [1, 2, 0])
result = run(
    "FROM $network RETURN COUNT(CELLS(1)), BETTI(1)",
    sources={"network": network},
)
assert result.values == (3, 1)
```

A query without `MATCH` returns one value for each `RETURN` expression.
`MATCH` iterates a finite cell selection and returns a sequence of rows.
The following query selects relations with more than two participants:

```python
from rexgraph import Relations, VertexTable

relations = Relations.from_supports(
    [[0, 1, 2], [1, 2]],
    vertices=VertexTable(("a", "b", "c")),
)
shared = RexGraph.from_relations(relations)
result = run(
    "FROM $graph MATCH e IN CELLS(1) WHERE ARITY(e) > 2 "
    "RETURN e.index, ARITY(e) ORDER BY e.index LIMIT 10",
    sources={"graph": shared},
)
assert result.values == (((0, 3),),)
```

`WHERE` requires a scalar boolean. `ORDER BY` requires compatible finite
scalars and uses a stable sort. `LIMIT` and `OFFSET` take nonnegative integers.
Selecting rows does not itself construct a subcomplex; restriction and
structural edits are explicit operators.

## Arithmetic and operator contracts

Integer and rational scalar expressions retain exact values:

```python
from fractions import Fraction

result = run("FROM $network RETURN SUM([1/3, 2/7]), MEAN([1, 2])",
             sources={"network": network}, exactness="exact")
assert result.values == (Fraction(13, 21), Fraction(3, 2))
```

The default `exactness="declared"` retains the arithmetic declared by each
source and operator. `exactness="exact"` checks that a supported finite read
query without MATCH returns no approximate or rounded numeric values. It
checks the result's arithmetic without converting a numerical algorithm. Exact channel and boundary actions
are requested through their operator arguments.

A Chain records coefficients for a combination of cells; a Cochain assigns
values to cells and evaluates those combinations. Boundary operators lower the
grade of a Chain, while coboundary operators raise the grade of a Cochain.
RCQL checks the grade and cell basis when combining typed values and operators,
including the identity and order of the cells. Explicit metrics determine
metric adjoints. The [operator reference](REFERENCE.md#operator-contracts)
covers Hodge, Green, Dirac, channel and metric operations.

## Capabilities

| Area | Queries and interfaces |
| --- | --- |
| Structure | Cell counts, supports, arity, boundaries and Betti numbers |
| Fields and channels | Typed actions, exact coefficients, Hodge components and channel measurements |
| Selection and edits | Restrictions, partitions, insertions and explicit structural changes |
| Registered data | Declared datasets, file catalogs and RCDB records |
| Plans and provenance | `EXPLAIN`, native plans and observed execution methods |
| Exact artifacts | Portable results and dependency checked query caches |
| Advanced declarations | Sheaf sections, type accessions, names and finite programs |
| Record mutations | Typed mutation requests with explicit publication and version guards |

Use `Executor` and `parse` to reuse a runtime, or the typed builder interfaces
to assemble queries in Python. The [API reference](REFERENCE.md) gives the
operator inventory, examples and requirements for advanced declarations.

## Registered sources and permissions

The application supplies source objects through `sources`. A declared dataset
combines a Core schema and relation mapping with a source file or stream.
A file catalog exposes selected roots under relative names, while an RCDB
binding exposes a supplied store. Query text resolves those registrations.
See [dataset and catalog examples](REFERENCE.md#local-artifacts).

A `BoundSource` can carry a `SourcePolicy`. Planning and execution enforce
operator permissions, including record history, identity and mutation
requirements. `EXPLAIN` checks the same type and permission contracts without
executing expression adapters; resolving a selected record can still read its
state to establish the source contract.

## Portable exact results

```python
from rcql import Result

payload = result.to_bytes()
restored = Result.from_bytes(payload)
assert restored.values == result.values
```

The result codec preserves exact values, plans, source references and
provenance. Graph results carry native sealed state, and supported Chain and
Cochain values retain their variance and basis. Decoding checks integrity
without rerunning the query. Values without a portable codec raise `TypeError`.
The default artifact limit is 64 MiB and can be configured with `max_bytes`.

System uses this codec for native result downloads, and the query cache uses
the same payload. See the [portable result reference](REFERENCE.md#portable-exact-results)
for the supported value inventory and cache identity rules.

## Record mutations

Mutation queries publish through RCDB's `commit_mutation` contract. They use
an explicit candidate state and can require an expected record version:

```text
FROM $db MUTATE "sample" SET state = $candidate COMMIT
```

The application binds `db` and `candidate` and supplies the required
permissions. `EXPLAIN` plans a mutation without publishing it. See the
[mutation reference](REFERENCE.md#mutations) and [RCDB guide](../rcdb/README.md)
for publication, history and concurrency behavior.

## Reference and testing

The [API reference](REFERENCE.md) covers operators and detailed mathematical
contracts. The [dependency guide](../DEPENDENCIES.md) lists development extras.
From the repository root, run `make test-rcql TEST_ARGS='-rs'` after installing
a development profile. RCQL is distributed under the [Apache License 2.0](LICENSE).

### Layer products and runtime revisions

`Layer`, `LayerInterface`, `LayerProduct`, `LayerSolution`, `LayerReadout` and
`LayerOperator` are finite retained declarations, available through the Python
API and the `LAYER_*` query operators. A product keeps named layer carriers,
participation roles and explicit compatibility maps. Affine products use native
section completion and retain the entire family and exact contradiction
certificates. `LAYER_VALUE` returns a prediction only when its readout is fixed
throughout that family. Local RCQL programs compile through the existing
`ProgramAssembly` and `ProgramFamily` machinery; unused latent directions remain
free. `LAYER_APPLY` returns a `ProgramResult` for affine programs, readable with
`PROGRAM_OUTPUT`, or a `TemporalModel` for function products.

With the optional `temporal` extra, whole partial functions, finite learning,
transfer, cover gluing and correlated context families execute in the installed
C++ RexTemporal runtime. Function layers declare their support and exact chart
scale and shift. Finite family products retain coupled alternatives and global
witnesses. Incomplete searches remain incomplete. Their projections are never
replaced by an independent Cartesian product or an arbitrary worker.

`LAYER_MODIFY(product,replacements,interfaces)` and
`FUNCTION_MODIFY(model,program,changes)` produce new declarations at runtime.
Earlier versions remain usable. Modified products retain their predecessor and
all contributor identities. They can be captured by ordinary finite RCQL
programs or used through named and recursive operations. Compilation and
`EXPLAIN` do not execute a target function. Applying an operator rechecks its
current sources, permissions and mathematical premises. This changes a callable
declaration; publishing a stored revision remains an explicit existing storage
operation.

`FUNCTION_SOURCE`, `FUNCTION_DECLARE`, `FUNCTION_PROGRAM`, `FUNCTION_APPLY`,
`FUNCTION_FIT`, `FUNCTION_TRANSFER`, `FUNCTION_JOINT`, `FUNCTION_INFO` and
`FUNCTION_VERIFY` expose the native runtime. Constructors accept closed finite
specifications; opcodes are `input`, `constant`, `derivative`, `pullback`, `add`,
`subtract`, `multiply`, `divide` and `restrict`. A constant program references a
retained constant-function object. Role bindings name actual source-local
function IDs. Training requires current `read` and `train` permissions and uses
explicit supplied observations (the NamedInputs profile). It does not infer
availability or training authority from a function domain or label.

`FUNCTION_FIELD(model,injection,grade,powers)` admits a single whole-domain
polynomial through an explicit scalar injection into an actual cell carrier.
Its ordered power axis remains a tensor axis, allowing native responses and
full cross moments. The operation refuses domain clipping, nonconstant
denominators, discarded powers and foreign carriers. Retain the model alongside
the derived field to retain its executable function source.

Layer values and models support `to_record`/`from_record`, `LAYER_RECORD`/
`LAYER_READ` and `FUNCTION_RECORD`/`FUNCTION_READ`. Portable query results also
retain these closed declarations. Decoding does not execute a native kernel or
restore authority; executing a reopened value requires current explicit source
bindings. Native C++ field and moment result owners reopen from their complete
checked archive with matching coordinate and implementation identities.

The canonical Core `RelationalRuntime` and these layer declarations exchange
complete native archives through `TemporalModel.from_relational`,
`TemporalModel.to_relational` and `FUNCTION_IMPORT`/`FUNCTION_EXPORT`. Conversion
checks the source and endpoint bytes and preserves source-local IDs. Exporting
a source-bound model as an unqualified packet is refused; keep its native record
and explicit contributors. Native execution requires the implementation receipt
recorded by the model; incompatible runtimes require an explicit migration.

The three admitted product profiles are affine interfaces, whole-function
covers and finite context families. Cross-profile maps require an explicit
adapter. Finite products preserve their native families; selecting an executable
worker still requires a complete qualified native fit. These APIs make no
unrestricted nonlinear or demand-loaded execution claim. See
[`examples/layer_product.py`](examples/layer_product.py) for a complete runtime
revision example.

## Ontology queries

Ontology operators retain complete `OntologyState` values through typed query
results, saved results and program composition. Supply a graph containing an
ontology declaration as the query source:

```python
from fractions import Fraction
from rexgraph import RexGraph
from rexgraph.ontology import OntologyState
from rexgraph.value import ExactTime
from rcql import Executor, parse

ontology = OntologyState.create(
    RexGraph.from_graph([0], [1]), identity="example:types", context="observation",
    known_at=ExactTime(Fraction(10)), signature={"types": ("relation",)},
    universe={"scope": "declared-source"},
)
result = Executor(sources={"types": ontology.record()}).execute(parse(
    "FROM $types LET o = ONTOLOGY_AT(ONTOLOGY_READ(), 10) "
    "RETURN ONTOLOGY_INFO(o), ONTOLOGY_EXPORT(o), o"
))
```

The [Core guide](../rexgraph/README.md#ontology-snapshots) also shows file storage.

| Operation | Result |
| --- | --- |
| `ONTOLOGY_READ(grade=1, index=0)` | Selected ontology in the source graph |
| `ONTOLOGY_IMPORT(archive)` / `ONTOLOGY_EXPORT(ontology)` | Whole RCBO snapshot / complete bytes |
| `ONTOLOGY_RECORD(ontology)` | Detached complete relational source |
| `ONTOLOGY_INFO(ontology)` / `ONTOLOGY_DECLARATION(ontology)` | Bounded inspection / complete declarations |
| `ONTOLOGY_AT(ontology, cutoff)` | Snapshot admitted by its declared knowledge horizon |
| `ONTOLOGY_REVISE(ontology, graph, changes, sources=None)` | Explicit child with parent and maps |
| `ONTOLOGY_VERIFY(ontology, cutoff, sources=None, parent=None)` | Explicit provider report |
| `ONTOLOGY_NATIVE_APPEND(ontology, component, frame, cutoff, sources=None)` | Child retaining an admitted native history frame |
| `ONTOLOGY_NATIVE_ADD(ontology, component, specification, cutoff, sources=None)` | Child with explicit native aspect/type additions |

Cutoffs accept exact integers, fractions or `ExactTime`. `ONTOLOGY_AT` checks
the declared horizon; actual publication time belongs to source selection.
Evidence dependencies are checked through existing source authority. Revisions
require read and train authority. `EXPLAIN` describes these requirements without
opening a provider or solving axioms. A provider's scoped report states what it
actually checked; retaining an ontology declaration grants no execution access.

The native revision operators require the optional C++ bridge and the actual
bound parent record and version. They return immutable children and retain the
parent as an evidence dependency. They do not publish records. Reusable programs
can carry these operations, while execution still requires read and train access;
Agent workers also require registration-owned write permission. See the
[native source guide](../bindings/cpp/README.md#native-source-revisions).
