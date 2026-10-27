# RexGraph Core

Core starts from the relationships an application needs to preserve. A pairwise
link and a measurement shared by several entities can each be one relation,
with its own participants and coefficients. Boundaries connect vertices at
grade 0, relations at grade 1 and faces at grade 2. Adding a face declares a
filling of its boundary; finding a loop alone does not add that filling.

These choices determine the homology, Hodge components and operator responses
computed below. Construct the complex first, then attach a signal and choose
the operation whose output answers the application question.

The Python API uses compiled Cython kernels for sparse storage and numerical
computation. Integer and rational paths retain exact coefficients and ranks;
numerical solvers expose their own tolerances and residuals.

## Installation

From the repository root:

```sh
make install-core
make python
```

Use `sh install.sh --profile core` when Make is unavailable. This installs the
Core library independently of Agent, RCDB, RCQL and System. `make install` includes
those application packages. The [installation guide](../README.md#installation)
describes the compiler, native libraries and optional dependencies.

## Constructing a complex

An ordinary graph uses one relation for each source and target pair:

```python
from rexgraph import RexGraph

rex = RexGraph.from_graph([0, 1, 2], [1, 2, 0])
print(rex.nV, rex.nE)              # 3, 3
print(tuple(rex.betti))           # (1, 1, 0)
```

The cycle has one connected component and one unfilled loop. A face must be
declared to fill that loop. `from_simplicial` accepts triangle vertex indices;
parallel relations require explicit face edges when the boundary is ambiguous.
`chain_valid` checks that composing successive boundaries gives zero.

For a shared relation, declare its full participant support, tail shares and
metric weight. An explicit vertex table can also retain isolated entities:

```python
from fractions import Fraction
from rexgraph import Relations, VertexTable

relations = Relations.from_supports(
    [[0, 1, 2, 3]],
    vertices=VertexTable(("head", "a", "b", "c", "isolate")),
    shares=[0, Fraction(1, 4), Fraction(1, 2), Fraction(1, 4)],
    weights=[Fraction(2, 3)],
    relation_ids=["assertion-1"],
)
shared = RexGraph.from_relations(relations)
assert shared.nV == 5 and shared.nE == 1
```

Declare participant shares, metric weights, signs, identifiers and attributes
separately. `Absent` records that a value was not supplied. A missing weight is
treated as one during calculation, while `relations.weight` retains its absence;
an explicitly supplied zero or one remains a present value.

| Constructor | Input |
| --- | --- |
| `from_graph` | Source and target arrays |
| `from_relations` | Declared participants, coefficients and identities |
| `from_hypergraph` | Compressed participant support arrays |
| `from_simplicial` | Pairwise relations and explicit triangles |
| `from_adjacency` | An adjacency matrix with a declared direction policy |

## Reading data

The declared readers separate file syntax from the meaning of its columns.
For a table, supply a `RecordSchema` and a `RelationSpec`; `construct` maps each
record into the common `Relations` carrier. Number rules specify whether a
decimal or binary source value remains exact. Numeric text is not inferred
from a field declared as a string.

The [root example](../README.md#loading-and-processing-data) reads a weighted
CSV table. The [I/O guide](io/README.md) covers JSON, JSONL, Arrow IPC, Parquet,
SQLite, XLSX and reusable dataset declarations.

## Signals and Hodge analysis

A signal assigns a value to each cell at one grade. For a relation signal,
Hodge decomposition returns gradient, curl and harmonic components:

```python
import numpy as np

signal = np.array([1.0, 0.0, -1.0])
gradient, curl, harmonic = rex.hodge(signal)
assert np.allclose(gradient + curl + harmonic, signal)
```

The gradient component comes from vertex potentials, the curl component from
face boundaries, and the harmonic component from the Hodge operator's kernel.
The cycle above has no face, so its upper operator and curl component are zero.
Declaring a filling changes which signals can be face boundaries, even if the
relation values stay the same. Interpret a component with its source and metric;
a nonzero residual alone does not label a relation as contradictory or ironic.

`Cochain`, `Chain`, `Field` and `GradedState` retain the cell grade and optional
ordered basis alongside values. This information lets boundary, coboundary
and Hodge operators check which cell space those values belong to. Numerical
Green actions solve a specified system without forming an inverse; supported
rational actions can be requested explicitly.

## GPU execution

Run `make install-gpu` from the repository root to install the platform with
automatically selected CUDA or ROCm dependencies. AMD installation includes the
custom HIP kernels. A Core installation can enable the same dependencies with
`sh install.sh --profile core --gpu auto`.

Vulkan supports float64 sparse products and the propagation and solver paths
that use them. Select it explicitly for a numerical operation:

```python
from rexgraph.scale_propagator import heat_apply

smoothed = heat_apply(rex.L1, signal, t=0.2, backend="vulkan")
```

Vulkan retains sparse coefficients on the GPU between products and performs
vector updates on the CPU. CUDA and ROCm propagation can retain the recurrence
on the device; their automatic selection uses a size threshold to account for
transfer costs. An explicit Vulkan request requires a visible hardware GPU with
float64 shader support and reports an error when that requirement is unmet.
`make verify-gpu GPU_BACKEND=vulkan` checks numerical agreement with CPU results.

## Capabilities

| Capability | Main interfaces |
| --- | --- |
| Homology and boundaries | `betti`, `chain_valid`, `graded_boundary`, `linear_operator` |
| Signal decomposition | `hodge`, `hodge_full`, typed Hodge operators |
| Structural channels | `channel_operator`, `structural_character`, `vertex_character` |
| Local responses | `coherence_response`, `character_response`, `local_context` |
| Diffusion and waves | `evolve_markov`, `evolve_schrodinger`, field evolution methods |
| Persistence | `filtration`, `persistence`, barcode and diagram methods |
| Selection and edits | `Selection`, `restrict`, `subgraph`, insertion and deletion |
| Temporal structure | `TemporalRex`, snapshot identities and lifecycle measurements |
| Models | [Numerical cochain models](flow/README.md) and `rexgraph.nn` |
| Storage | [Native file formats and declared readers](io/README.md) |

The four structural channels are T, G, F and C. T measures signed boundary
incidence, G measures unsigned overlap with relation weights, F measures the
discrepancy between signed and unsigned incidence, and C measures coparticipation.
The geometry G channel belongs to the down sector, which reads a relation's
vertex boundary. Curl uses the up sector, which reads its face boundary.
The up operator at grade k and down operator at grade k+1 share their nonzero
spectrum and generally have different shapes.

Sparse operator APIs retain boundary factors and materialize matrices when
requested. Dense matrix properties and spectral methods remain available for
explicit numerical analysis. Choose those interfaces according to the size of
the complex and the required output. The [API reference](REFERENCE.md) lists
the methods, shapes and arithmetic contracts.

## Saving state and selecting cells

```python
from rexgraph.io import load_rcbd, save_rcbd

save_rcbd("cycle.rcbd", rex)
restored = load_rcbd("cycle.rcbd")
assert tuple(restored.betti) == tuple(rex.betti)
```

Native state retains declared coefficients, identities and registered
components. Table exports have their own schemas and do not automatically
represent a complete saved complex. Copies and restrictions use the same
component transport; a restriction drops an upper cell when its boundary loses
a participant. `Selection` and `restrict` provide explicit grade selections
with maps back to the original basis. See the
[state and selection contracts](REFERENCE.md#declared-data-and-exact-io).

## Exact native models

The optional `rexgraph_cpp` provider executes whole field programs with exact
source and map declarations. Install it using the
[bridge guide](../bindings/cpp/README.md#installation). These models retain
intermediate fields and unresolved alternatives; the numerical cochain models
listed above have separate optimizers and checkpoint formats.

An available finite map retains symbolic alternatives and complete fixed or affine
field targets. Its Core model stores the original literal source, immutable map
archive and the ordered target model checkpoints:

```python
from rexgraph.cpp_relational_finite_field import (
    cpp_available_finite_field_model, finite_field_query, read_finite_field_result,
)
from rexgraph.model_runtime import infer_model

state = cpp_available_finite_field_model(
    literal_source, available_map=available_map, targets=target_models,
    execution_context=execution_context,
)
output = infer_model(
    state, literal_source, finite_field_query(state),
    execution_context=execution_context,
)
family = read_finite_field_result(output, execution_context=execution_context)
branches = family.select(0, 1, execution_context)
```

The instance executes its original raw rule inputs. It does not rebind new data
into the stored map. `evaluate_finite_field` selects an interpreted branch using
its full free vector, the independent global parameter setting and a separate
input specific Prospective choice receipt. Fixed branches use an empty vector and
setting key zero. `finite_field_stalks` reports the actual target source after an
explicit completion. Ordinary training, undeclared source transport and promotion
into the existing local solve RHS are unsupported.

RCQL exposes `MODEL_INIT` with adapter `cpp_relational_finite_field`,
`MODEL_FINITE_BRANCHES`, `MODEL_FINITE_COMPLETE`, and the common model inspection,
record and stalk operations. Admitted operations require the current native
execution context. Passive model/value packing does not grant execution authority.

Admitted finite operators reuse an original finite model with independently
supplied input instances. Each instance retains its declared source transport,
complete input fields and publication receipts:

```python
from rexgraph.cpp_relational_finite_operator import (
    cpp_available_finite_operator_model, finite_operator_query,
    read_finite_operator_result,
)
from rexgraph.model_runtime import infer_model

model = cpp_available_finite_operator_model(
    original_source, reference_state=original_finite_model,
    available_operator=admitted_operator, execution_context=context,
)
query = finite_operator_query(model, admitted_input_instance,
                              execution_context=context)
output = infer_model(model, original_source, query, execution_context=context)
family = read_finite_operator_result(output, state=model,
                                    execution_context=context)
branches = family.select(0, family.inspect(context)["branch_count"], context)
```

The model keeps its original source and checkpoint. The output records the
actual input source and instance separately and retains every unresolved
branch and whole field. Affine completion requires a published global parameter
setting and a separate prospective choice for the input. Fixed completion uses
empty coordinates and parameter setting key zero. Ordinary training and
implicit source transport are unsupported for the operator checkpoint.

## Examples and reference

See the [compiled kernel guide](core/README.md), [API reference](REFERENCE.md)
and [dependency guide](../DEPENDENCIES.md) for supported operations, file readers
and optional analysis packages. Core is distributed under the
[Apache License 2.0](../LICENSE).

## Ontology snapshots

`OntologyState` retains a complete relational source together with its signature,
universe, ordered axioms, interpretations and explicit evidence references.
Snapshots are immutable. Revisions retain the selected parent, the declared
comparison maps and an exact knowledge horizon. Declaring axioms or maps does
not prove them; mathematical validation requires an explicitly installed provider.

```python
from fractions import Fraction
from rexgraph import RexGraph
from rexgraph.ontology import OntologyState
from rexgraph.value import ExactTime
from rexgraph.io import save_ontology, load_ontology

source = RexGraph.from_graph([0], [1])
ontology = OntologyState.create(
    source, identity="example:types", context="observation",
    known_at=ExactTime(Fraction(10)),
    signature={"types": ("relation",)},
    universe={"scope": "declared-source"},
)
save_ontology("types.rcbo", ontology)
restored = load_ontology("types.rcbo")
assert restored.digest == ontology.digest
assert restored.to_bytes() == ontology.to_bytes()
```

The [RCBO format](io/README.md#ontology-files) preserves the selected ontology
anchor and whole source. Ordinary RCBF, RCBD and safetensors containers can carry
the same graph declaration through `ontology.record()`. `require_horizon` checks
the declared information horizon; an external store's publication cutoff and
access policy are separate checks.

The optional `rexgraph.native_ontology` provider decodes explicitly enclosed
native source histories and checks exact parent prefixes through the C++ bridge.
Its report distinguishes this enclosure from an unproved graph-to-native map,
arbitrary axiom verification and current native availability. Files never load
provider code automatically.

`append_native_ontology` and `add_native_ontology_declarations` create immutable
children for an existing native component. Both require a checked `FieldSource`
identifying the actual parent record and positive version, plus an exact knowledge
horizon. The child retains that parent dependency and complete source history.
Native declaration additions use explicit aspect fills or complete occurrence
assignments and declared exact relation types. See the
[native source guide](../bindings/cpp/README.md#native-source-revisions).
