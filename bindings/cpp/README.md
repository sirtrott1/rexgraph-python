# RexGraph C++ Python bindings

The `rexgraph_cpp` package connects Python applications to the exact source,
field, storage and model operations in RexGraphCPP. Python supplies declarations
and chooses what to execute. The C++ library constructs the native objects,
applies sparse operators and solves integer and rational equations.

A field value is useful only with the space it belongs to. The bridge therefore
keeps its source owner, carrier, support and metric with its coefficients. A
prepared operation can be reused on that source, and its result owns the objects
needed to inspect it after the call returns. Equal array shapes do not authorize
a calculation on a different source; crossing sources requires a declared map.

The sections below follow construction through field actions, model execution
and storage. Complete intermediate fields remain accessible before readout. If
the equations leave several possibilities, the result keeps the alternatives
and free parameter directions for the application to inspect or constrain.

## Installation

Install the `rexgraph-cpp` 0.3.0 SDK with CMake, then build the bindings using its
installation prefix. The build requires a C++20 compiler, GMP, GMPXX, Python
development headers and Cython.

```sh
python -m pip install ./bindings/cpp \
  --config-settings=cmake.define.CMAKE_PREFIX_PATH=/path/to/installed/sdk
```

The source checkout also provides the equivalent Make target:

```sh
make install-cpp CPP_SDK_PREFIX=/path/to/installed/sdk
```

Core remains usable without the bindings. Native models require their recorded
provider and compatible implementation when constructed or restored. Keep the
matching runtime for existing model records; reading archive bytes and executing
a stored wrapper have separate compatibility checks.

The bridge and native C++ library use Apache License 2.0. This distribution
includes the [license](LICENSE) and [copyright notice](NOTICE). Third-party
components retain their own licenses.

When the bindings are loaded, model content checks, metadata copying and payload
codecs use compiled storage paths. Tensor checks compare complete owned byte
images, including exact fractions and source declarations. Canonical value
encoding produces the same bytes with or without the bindings, so installing
the bridge does not change the identity of an existing stored value.

## Source layout

The Python and Cython package is the top level `rexgraph_cpp/` directory beside
`rexgraph/`. Its C++ adapters, build configuration and tests are under
`bindings/cpp/`. The independent `rexgraph-cpp-bindings` wheel installs only the
bridge package; Core and the application packages are installed separately.
The bridge source distribution includes the complete `rexgraph_cpp/` package
and can be built after extraction using the same installed SDK configuration.

## Source construction

Begin with the source declaration when the application has relations to retain.
Use [field programs](#whole-field-programs) to compose maps over whole fields,
and [available whole field learning](#available-whole-field-learning) when
published evidence constrains those maps. The
[finite source interfaces](#finite-sources-and-interpretations) keep symbolic
alternatives without assigning arbitrary numerical coordinates to them.

`NativeSource.from_declaration` creates a canonical archive from exact Python
declarations. The [source and training example](examples/source_models.py)
constructs relations and parameter fields, trains integer and rational models,
then saves, restores and resumes a model record.

```sh
python bindings/cpp/examples/source_models.py /path/to/output
```

```python
from pathlib import Path
from rexgraph_cpp import NativeSource

source = NativeSource.from_declaration(declaration)
Path("source.rfcxs").write_bytes(source.archive)
carrier = source.carrier(11)
```

`NativeSource(archive, max_dimension=..., max_records=...)` sets explicit
reader bounds for logical tensor sizes and encoded records. Both default to
1,000,000. For a learned pattern bank, `stalk_pattern_bank_source` in
`rexgraph.cpp_stalk_pattern_model` uses the bank's declared `limits`.
Model storage and local bank selection retain those limits.

The `declared-exact-source-v1` profile requires `namespace`, `semantic_scale`,
`max_grade`, `types`, `witnesses`, `relations`, `carriers`, `fields` and `metrics`.
Types declare names and arity bounds. Relations retain grade, key, type index
and ordered occurrences with explicit keys, lower identities, existence,
orientation and exact magnitudes. Grade 1 occurrences refer to witness
identities; higher grades refer to previously declared lower relation keys.
Repeated and inactive occurrences remain present. Inactive incidence coefficients
are stored as zero. The native builder validates
arity and higher grade closure. Explicit witness declarations preserve isolated
witnesses; `max_grade` preserves empty grades.

`declared-role-source-v1` additionally accepts named `role_aspects`. Each
occurrence's `aspects` tuple selects its role indices in that declaration.

This profile supports linear `source_grade` and `explicit` carriers, sparse
fields and diagonal metrics. A field's `support=None` declares full support,
including zeros omitted from its entries. An explicit support tuple declares
only those coordinates. Coefficients accept integers and `Fraction` values.
Unknown declaration fields and floating coefficients are rejected. Construction
starts a new temporal namespace at sequence zero; it does not append to an
existing source or infer a signature from a Python graph.

Construction accepts `max_records`, `max_slots`, `max_integer_bits` and
`max_bytes`. The defaults are 1,000,000 records, 16,000,000 reference slots,
16,777,216 aggregate coefficient bits and a 256 MiB archive. The archive byte
bound is checked after serialization. These are declaration and archive bounds,
not total process memory limits.

The archive codec also bounds each tensor's logical domain to 1,000,000
coordinates. Stored diagonal metrics use a square matrix domain, so this
construction profile permits at most 1,000 cells per grade with declared
diagonal metrics. Named unit metrics do not create that matrix domain.

`literal_token_chart` constructs byte and token interval sources in native code.
Supply the original bytes, Unicode words and ordered byte spans. It returns the
token source, byte source and their complete chain map. Use `retain_family=True`
to also return the sparse grade maps for storage or replay.

```python
from rexgraph_cpp import literal_token_chart

tokens, raw, mapping, family = literal_token_chart(
    "猫 dog".encode("utf-8"), ("猫", "dog"), ((0, 3), (4, 7)),
    retain_family=True,
)
```

The profile admits up to 8192 bytes and 128 supplied words. It retains bytes
outside the spans and distinct token cuts with coincident byte images.
`max_bytes` bounds each archive; `limits` bounds chain records, work and integer
bits. Token boundaries are supplied by the caller.

`literal_whitespace_chart(raw_bytes)` supplies spans using bytes 9 through 13
and byte 32 as separators. It returns the token source, byte source, complete
map and ordered spans. With `retain_family=True`, it also returns the grade
maps. Punctuation, case and bytes outside ASCII stay unchanged; empty or whitespace
only input rejects. The same size and resource bounds apply.
The byte source uses `namespace`; the token source uses `namespace + 1`.

`literal_whitespace_spans(raw_bytes)` returns the same intervals without building
sources or maps. Use it to prepare ordered participants before creating a source.

## Relational channels

The scalar channel model returns T, raw G, F and C fields on the original
grade 1 relations. Coefficients and relation weights accept integers and
`Fraction` values. The field metric is unit. Each relation requires distinct
participants; inactive occurrences and supported zeros remain in the source.

```python
from rexgraph.cpp_channels import cpp_channel_model, relational_channel_query
from rexgraph.model_runtime import infer_model

model = cpp_channel_model(source, coparticipation="share")
query = relational_channel_query(model, relation_values)
output = infer_model(model, source, query)
fields = output.tensor()
```

The output channel axis is `("T", "G", "F", "C")`. Set `coparticipation="count"`
for support counts or `trace_normalized=True` for trace normalization. Zero
trace channels retain zero action. This model declares fixed actions; learning
requires a separate model provider.

## Context correspondences

`cpp_correspondence_model` retains a context chart with two spans, its exact byte
source and its declared pattern source. Complete templates for chain maps and exact
integer counts specify the interpretation. An unobserved context selects the
explicit `unresolved` template. Span selection and pattern topology are supplied.
The fixed version accepts supplied counts and declares no training or transport.

```python
from rexgraph.cpp_correspondence import (
    cpp_correspondence_model, cpp_correspondence_input,
)
from rexgraph.model_runtime import infer_model
from rexgraph.io.model_record import model_record, save_model

state = cpp_correspondence_model(chart, raw_source, pattern_source, **declaration)
inputs = cpp_correspondence_input(state, chart, values, side="pattern", integer=True)
output = infer_model(state, chart, inputs)
save_model("context.rcbd", model_record(chart, state))
```

The declaration supplies complete grade carriers, byte spans and literal bytes,
map templates, labels, the unresolved label index and counts. The raw source
stores UTF-8 bytes as a fully supported grade 1 field, including zeros.
`side="raw"` applies the provenance map. Pattern outputs retain rational values
and both dependency references. `output.tensor()` binds the field to the selected
target and retains the observed chart as evidence. Integer execution uses checked int64 arithmetic
with a retained common denominator and requires unit native lattice scales.
The chart declares a coarser semantic scale than the pattern source. Fractional
template coefficients remain exact through the shared denominator.
Loading validates source archives, literal spans, carriers and every map grade.

### Context count training

`cpp_context_catalog` fits integer counts for declared UTF-8 mention, positional
role and context columns. Each column occupies a declared grade 2 coordinate.
Annotations retain unfitted correspondence models, chart archives, exact labels,
partition and availability times. Each annotation batch creates one immutable
child. Reusing a sample identity, fitting evaluation labels or selecting unavailable
parameters rejects. Frequencies remain rational; unobserved columns select the
declared unresolved interpretation.

```python
from rexgraph.cpp_context_catalog import (
    cpp_context_catalog, cpp_context_annotation, cpp_context_training_batch,
    cpp_learned_correspondence,
)
from rexgraph.model_runtime import train_model

catalog = cpp_context_catalog(
    parameter_source, pattern_source, carrier=parameter_carrier,
    features=features, labels=labels, unresolved=unresolved,
)
sample = cpp_context_annotation(
    "example-1", unfitted_context, observed_chart, feature=0, label=0,
    event_time=10, knowledge_time=11, label_available_time=12,
)
batch = cpp_context_training_batch(
    catalog, parameter_source, (sample,), event_cutoff=10, knowledge_cutoff=12,
)
fitted = train_model(catalog, parameter_source, batch)
state = cpp_learned_correspondence(
    fitted, parameter_source, query_chart, query_raw, pattern_source,
    event_cutoff=20, knowledge_cutoff=20, **query_declaration,
)
```

`features` contains `(mention_bytes, "left" or "right", context_bytes)` tuples.
`query_declaration` supplies the usual complete carriers, scope, templates and
interpretation labels; counts come from the selected catalog. Version 2
correspondences retain the complete fitted checkpoint and parameter source.
Their outputs and tensor fields retain the parameter reference beside chart,
byte and pattern provenance. Unknown feature keys select the unresolved template.
Segmentation, vocabulary and pattern topology remain supplied declarations.

Catalogs support the existing model records, RCQL model creation, training and
inference, Agent prediction workers and System serialization. RCBD and safetensors
retain annotations and exact parent checkpoints and permit continued catalog
training after restoration. The default history byte limit is 64 MiB; complete
history verification admits at most four times that limit in serialized prefixes.
The finite catalog bounds are 20,000 columns, 64 labels, one million annotations
and 1,024 updates. Byte and verification limits can bind before those counts.

## Algebraic archives

`NativeAlgebraicArchive` loads canonical RFCXA archives written by the C++
`AlgebraicArchive` codec. Record keys select declared vectors, covectors, maps,
measurement forms and certified metrics.

```python
from pathlib import Path
from rexgraph_cpp import NativeAlgebraicArchive
from rexgraph.io import save, load
from rexgraph.io.algebraic_record import algebraic_record, read_algebraic_record

archive = NativeAlgebraicArchive(Path("fields.rfcxa").read_bytes())
value = archive.record(10)
result = archive.record(30).apply(value)
solved = archive.record(50).solve(value)
save("fields.rcbd", algebraic_record(archive))
restored = read_algebraic_record(load("fields.rcbd"))
```

The keys must exist with the declared record kinds. `archive.manifest` lists
records, field certificates, selected embeddings and source sizes. Each load
creates fresh native owners. Operations require the actual declared field and
carrier owners; equal bytes in another archive do not identify them.

`coefficients` returns row major coefficient blocks as tuples of `Fraction`
values in the field's polynomial basis. `shape` uses module ranks;
`field_degree` in a preview gives the rational coordinates per coefficient.
`preview(max_values)` bounds the number of coefficient blocks displayed. System
previews retain these exact coordinates and their field and carrier declarations.

Maps and forms support `apply`; certified metrics also support `solve`.
Covectors support `evaluate`, and forms and metrics support `quadrance` and
`as_covector`. Named reconstruction pairs support `archive.accession`,
`archive.reconstruct` and `archive.project`. Each archive retains at most one
validated reconstruction pair for repeated calls. Resource overrides use the names
returned by `NativeAlgebraicArchive.default_limits()`, for example
`limits={"max_file_bytes": 8 * 1024 * 1024}`. Core containers and RCDB retain
the complete native archive; restoration rechecks its declarations and proofs.

`archive.vector(record_key, coefficients, carrier_side="source")` constructs
an exact vector on that record's declared module. Each module coordinate
requires a complete polynomial coefficient block containing integers or
`Fraction` values. Use `carrier_side="target"` to select a map's output carrier.

Use `cpp_algebraic_model` to run these actions through the model interface.

```python
from rexgraph.cpp_algebraic import cpp_algebraic_model, cpp_algebraic_input
from rexgraph.model_runtime import infer_model
from rexgraph.io.model_record import model_record, save_model, load_model, read_model_record

state = cpp_algebraic_model(archive, operator_key=30, input_key=10, operation="apply")
inputs = cpp_algebraic_input(state, archive, archive.record(10).coefficients)
output = infer_model(state, archive, inputs)
save_model("algebraic-model.rcbd", model_record(archive, state))
saved, owner = read_model_record(load_model("algebraic-model.rcbd"))
output = infer_model(saved, owner, inputs)
```

The model retains its selected action, input carrier and polynomial basis.
Outputs use `arithmetic="algebraic"` and retain exact coefficient blocks,
native value kinds and execution receipts. A retained vector output can be
passed to `cpp_algebraic_input` for explicit composition on the same declared
carrier. Inference also accepts an owning vector from the current archive.

RCQL `MODEL_INIT("cpp_algebraic", configuration)` accepts the factory's keyword
arguments over an algebraic archive or model record. `MODEL_INFER` retains the
algebraic contract; `MODEL_VALUES` returns its rational polynomial coordinates.
These fixed actions support inference and persistence. Training, parameter
transport, model histories and automatic floating conversion are unsupported.
`read_model_record`, `save_model` and `load_model` accept `algebraic_limits`
for reader admission, intersected with the saved model's native limits.

## Algebraic archive graphs

`NativeAlgebraicArchiveGraph` loads canonical RFCXG files. Its named tables
retain shared fields, bundles, metric domains, archives and complete comparisons.
`graph.archive(key)` selects a view on the actual graph owners. Separate loads
create separate owners even when their bytes are equal.

```python
from rexgraph_cpp import NativeAlgebraicArchiveGraph
from rexgraph.cpp_algebraic_graph import cpp_algebraic_compare

graph = NativeAlgebraicArchiveGraph(Path("comparisons.rfcxg").read_bytes())
source = graph.archive(1)
target = graph.archive(2)
state = cpp_algebraic_model(source, operator_key=20, input_key=10)
output = infer_model(state, source)
transferred = cpp_algebraic_compare(
    output, source, target, kind="carrier", key=1, carrier_record=10,
)
target_state = cpp_algebraic_model(target, operator_key=20, input_key=10)
inputs = cpp_algebraic_input(target_state, target, transferred)
prediction = infer_model(target_state, target, inputs)
save_model("graph-model.rcbd", model_record(source, state))
```

The selected keys must match the graph's declarations. `graph.comparison(kind,
key, max_values)` previews the complete map's dimensions and bounded coefficients.
`graph.compare(kind, key, value, target_archive=key)` applies scalar, carrier or
certified measurement comparisons to owning values. Measurement comparisons also
support `operation="quadrance"`. Reconstruction comparisons require an explicit
`accession`, `reconstruct` or `project` operation. Domain comparisons are receipts
for selected roots and involutions. They do not supply a carrier action.

`cpp_algebraic_compare` transfers retained scalar or vector model outputs through
named scalar, carrier or measurement comparisons. A vector requires its source
carrier record. Transfers retain their origin and graph reference; they do not
transport trainable parameters or infer inverses. A selected archive's record
retains the complete graph, including unused registered owners and comparisons.
`algebraic_graph_record` and `read_algebraic_graph_record` store and restore a
whole graph through the same Core formats and RCDB stores.

Graph admission uses the names from `NativeAlgebraicArchiveGraph.default_limits()`.
Archive limits have the `archive.` prefix; graph byte, owner, edge and aggregate
coordinate limits are separate. A graph model retains both policies.
`algebraic_limits` on model readers accepts this graph policy for graph models.

## Algebraic queries

Publish an owning archive with explicit namespace and sequence labels. Queries
accept `vector`, `covector`, `map`, `form` or `metric` records and exact `eq` or
`ne` predicates on complete polynomial coordinates.

```python
from rexgraph_cpp import NativeAlgebraicSnapshot, NativeAlgebraicTensorWindow
from rexgraph.cpp_algebraic_query import (
    prepare_algebraic_query, algebraic_query_record,
    read_algebraic_observation_record, read_algebraic_snapshot_record,
)
from rexgraph.io import save, load

snapshot = NativeAlgebraicSnapshot(archive, namespace=4001, sequence=5)
availability = (4001, 5)
query = prepare_algebraic_query(snapshot, {"kind": "vector"}, availability=availability)
result = query.execute(availability=availability, cpu_workers=2)
save("query.rcbd", algebraic_query_record(result))
record = load("query.rcbd")
observed = read_algebraic_observation_record(record)
restored = read_algebraic_snapshot_record(record)

pin = restored.record(10)
window = NativeAlgebraicTensorWindow(pin, (0,), pin.domain, availability=availability)
total = window.sum(availability=availability)
```

Record keys and window bounds must match the archive. An empty window has no
sum; supported zero entries remain present. Namespace and sequence label this
publication without asserting native history or event times. Restored snapshots
have fresh publication owners.

Use `source_key`, `target_key` and `all_of` in the query request to select named
carriers and exact coordinate predicates. Each predicate has `coordinate`,
`operation` and `literal` fields. A literal supplies one integer or `Fraction`
per polynomial coordinate. `execution_limits` names `records`, `rational_slots`,
`sparse_entries`, `scalar_work` and `integer_bits`. Execution rechecks these
limits, availability and cancellation. `NativeAlgebraicQueryCache(max_records)`
retains results for the same prepared owner; its bound counts result pins.

`algebraic_snapshot_record` supplies a labelled source record for RCQL.
`ALGEBRAIC_QUERY(request, max_sequence)` and
`ALGEBRAIC_WINDOW(key, lower, upper, max_sequence)` return the same exact
observations through Agent query execution. Observation records use existing
Core formats and RCDB stores. Reading saved observations requires Core;
restoring owners and executing queries also requires the bindings.

## Algebraic history

Read a version 2 algebraic archive channel from a native temporal series.
The three channel names identify its namespace, registered type and value.
The sequence selects an actual retained endpoint.

```python
from pathlib import Path
from rexgraph_cpp import NativeSource, NativeAlgebraicSnapshot
from rexgraph.io import save, load
from rexgraph.cpp_algebraic_history import algebraic_history_record, algebraic_history_at
from rexgraph.cpp_algebraic_query import read_algebraic_snapshot_record

source = NativeSource(Path("history.rfcxs").read_bytes(), max_bytes=64 * 1024 * 1024)
channel = (701, 1, 100)
snapshot = NativeAlgebraicSnapshot.from_history(source, channel, sequence=41)
save("history.rcbd", algebraic_history_record(source, channel, sequence=41))
record = load("history.rcbd")
restored = read_algebraic_snapshot_record(record, history_max_bytes=64 * 1024 * 1024)
earlier = algebraic_history_at(record, 40)
```

The channel names and sequences must exist in the archive. A history record
retains the complete native prefix ending at the selected sequence and its
algebraic image. Restoring rechecks the channel registry, archive proofs and
source binding. Later frames are absent from that prefix. Each restore creates
fresh native field and carrier owners.

`ALGEBRAIC_AT(sequence)` returns the selected prefix as a source record.
Bind that result for subsequent `ALGEBRAIC_QUERY` or `ALGEBRAIC_WINDOW` calls.
`ALGEBRAIC_HISTORY_INFO()` reads the retained prefix and channel declaration.
Saved observations include that declaration and remain readable with Core alone.

`history_max_bytes` bounds the complete prefix independently of the algebraic
archive limits. Readers intersect this bound with the saved source's byte limit.
Sequence numbers identify native publications; they do not supply event times
or parameter transport. This reader supports RFCXA channels, including chunked
payloads. Legacy integer channels require native migration before reading.
RFCXG history channels and durable history writing are unsupported.

## Phase and fold words

`NativePhaseFoldArchive` loads canonical RFCXW operation words. The supported
kinds are rational and algebraic phase histories, rational fold histories,
and rational and algebraic cyclic walks. Coordinates use integers and
`Fraction` values. Fold endpoints contain row major 4 by 4 panel frames;
algebraic phases retain their polynomial basis and involution.

```python
from pathlib import Path
from rexgraph_cpp import NativePhaseFoldArchive
from rexgraph.cpp_phase_fold import (
    phase_fold_record, read_phase_fold_record,
    phase_fold_observation_record, read_phase_fold_observation_record,
)
from rexgraph.io import save, load

word = NativePhaseFoldArchive(Path("operations.rfcxw").read_bytes())
save("operations.rcbd", phase_fold_record(word))
restored = read_phase_fold_record(load("operations.rcbd"))
save("prefix.rcbd", phase_fold_observation_record(restored, 2))
observation = read_phase_fold_observation_record(load("prefix.rcbd"))
```

`observed_prefix(count)` selects a logical operation prefix. It drops later
operations and all pending replacements. Observation records retain only the
complete selected word and its exact endpoint. Reading a saved observation
does not require the bindings or Torch. `observe(max_coordinates=65536)`
requires room for the complete endpoint, operation coordinates, domain
declarations and cyclic lift positions.

Phase and fold histories support `propose(key, position, coordinates,
active=True)` and `apply(key)`. A proposal leaves the original word unchanged.
Applying it creates a branch through native suffix replay and removes prior
proposals. Repeated identities, inactive operations, ordered moving panel
groups and cyclic backtracking remain present. Cyclic walks do not declare a
replacement profile.

`NativePhaseFoldArchive.from_history(source, channel, sequence=revision)` reads
a declared RFCXW channel from a complete retained native publication prefix.
The three channel names are explicit. `phase_fold_history_record` saves both
the word and that prefix; `phase_fold_history_at` selects an earlier retained
publication. Publication sequences and logical operation counts are separate.
These words do not declare physical time or model parameter continuation.

RCQL provides `PHASE_FOLD_INFO`, `PHASE_FOLD_PREFIX`, `PHASE_FOLD_OBSERVE`,
`PHASE_FOLD_PROPOSE`, `PHASE_FOLD_APPLY` and `PHASE_FOLD_AT`. Records use the
existing Core formats, RCDB stores and `copy_record`. Reader overrides use the
35 names returned by `NativePhaseFoldArchive.default_limits()`; restoration
intersects saved and reader limits. Native prefix bytes have an independent
`history_max_bytes` admission.

## Exact field models

Load a canonical native temporal series archive. Select its declared carriers,
metric tensors and input field by key.

```python
from pathlib import Path
from rexgraph_cpp import NativeSource
from rexgraph.cpp_model import cpp_model
from rexgraph.model_runtime import infer_model
from rexgraph.io.model_record import model_record, save_model, load_model, read_model_record

source = NativeSource(Path("source.rfcxs").read_bytes())
state = cpp_model(
    source, (10, 11, 12), operation="hodge", grade=1,
    metric_keys=(2000, 2001, 2002), metric_selection="diagonal",
)
output = infer_model(state, source, source.field(1001))
save_model("model.rcbd", model_record(source, state))

restored, source = read_model_record(load_model("model.rcbd"))
output = infer_model(restored, source, source.field(1001))
```

The keys in this example must exist in the archive. `source.carriers`,
`source.tensors` and `source.identity` expose its declarations. Output values
are Python `Fraction` values in declared primary coordinate order. Its
`execution_receipt` records the native field, carriers, profile, implementation
and resource requirements.

Supported actions are `boundary`, `adjoint`, `hodge_down`, `hodge_up`, `hodge`,
`green` and `harmonic`. Use `metric_selection="unit"` without metric keys for
the named unit metric. Declared diagonal metrics support finite actions.
General positive metrics require `metric_selection="positive"` and
`profile="global"`.

Green and harmonic actions require `profile="global"`. The global profile has
a default dimension limit of 64 per grade. Parameters accept integers and
`Fraction` values. Floating inputs are rejected. This profile executes stored
native fields with full retained grade support, including supported zeros.
Prepared execution requires nonempty retained grades. It does not provide
training or accept arbitrary feature arrays.

### Rational batch inference

Finite field models and exact channel models accept ordered stored fields through
`cpp_rational_batch_input`. Each field must have the model's declared input
carrier and full support. Sample coordinates name the columns independently.

```python
from rexgraph.cpp_model import cpp_rational_batch_input
from rexgraph.type_accession import CoordinateSpace

samples = CoordinateSpace("observations", ("first", "second"))
inputs = cpp_rational_batch_input(state, source, (1001, 1001), samples)
output = infer_model(state, source, inputs)
```

Rows follow the result's source grade and columns follow the sample coordinates.
Repeated field keys remain separate samples. Values stay exact `Fraction` values.
Models, captured inputs and outputs use the existing model records and codecs.
RCQL `MODEL_INFER`, Agent callable workers and System results use the same route.
The helper also accepts a fitted `cpp_metric` state without changing its training
history or sample cursor.

Direct calls use `PreparedField.execute_batch(fields, operation="hodge", grade=1)`
or `NativeTrainer.forward_batch(fields)`. Native execution releases the GIL and
serializes calls sharing a workspace. It reuses rational boundary and metric
scratch and checks cancellation between contractions and metric channels.
Growth admits both the retained workspace and its replacement before allocation.
Slot counts bound rational containers rather than GMP heap bytes. Source accession
work is separately capped at 4,000,000 units for these handles; larger requests
reject even when the selected execution limits permit them.

This batch profile supports named unit or declared diagonal metrics and finite
boundary, metric adjoint and Hodge actions. Global Green and harmonic profiles
require their existing scalar calls. Training retains its ordered serial updates.

The [rational batch example](examples/rational_batches.py) compares complete
binding and Core inference with scalar calls. Source, model and input setup are
reported separately. It uses stored field samples and five scalar metric model
parameters; it does not train a model.

```sh
python bindings/cpp/examples/rational_batches.py --cells 512 \
  --batches 1 4 8 --repeats 3 --output rational-timings.json
```

## Integer networks

Declare each layer's ordered channels, integral gains, scalar bias and
activation. Inputs use the source's primary coordinate order.

```python
from rexgraph.cpp_learning import cpp_integer_model, cpp_integer_input

state = cpp_integer_model(source, 11, [
    {"channels": ("identity", "hodge_down", "hodge_up"),
     "gains": (2, -1, 1), "bias": 0, "activation": "relu"},
])
inputs = cpp_integer_input(state, source, (1, -2, 3))
output = infer_model(state, source, inputs)
```

Channels are `identity`, `hodge_down`, `hodge_up` and `hodge`. Activations are
`identity`, `relu` and `sign`; sign maps zero to one. These channels use the
integral source lattice and coordinate transpose, independently of rational
metric fields. Arithmetic is checked signed 64 bit arithmetic. Overflow raises
an error; there is no rounding or saturation. The model envelope stores exact
rational outputs. This profile supports inference and durable model records.

### Batch inference

`cpp_integer_batch_input` captures independent field samples in a contiguous
native int64 matrix. Rows follow the model's source coordinates. Columns use
a separately named, ordered sample space.

```python
import numpy as np
from rexgraph.cpp_learning import cpp_integer_batch_input
from rexgraph.type_accession import CoordinateSpace

samples = CoordinateSpace("observations", ("first", "second"))
values = np.array([[1, 2], [-2, -3], [3, 0]], dtype=np.int64)
inputs = cpp_integer_batch_input(state, source, values, samples)
output = infer_model(state, source, inputs)
```

The input captures its own coefficient storage. Batch outputs retain exact
int64 coefficients, the sample axis and the normal model/source seals.
`output.tensor()` creates the rational tensor view when needed. Models, captured
inputs and outputs use the existing RCBD, safetensors and RCDB codecs. RCQL
`MODEL_INFER` and Agent callable workers accept the same input record.

The native kernel traverses retained lattice occurrences across sample columns.
It reuses boundary and channel buffers and caches down/up responses within each
layer. Ordered gains, duplicate occurrences, inactive attachments, checked
overflow and activation rules agree with scalar inference. These operations
use lattice coordinate transpose independently of rational metric fields.

Prepared networks retain the largest admitted batch workspace. Growth charges
both the old workspace and its replacement before allocation. The selected
resource limits bound input/output slots, retained scratch and work. The eight
entry prepared cache bounds entry count rather than total process memory.
Empty cell or sample axes, floating values, other integer widths, foreign byte
order and strided buffers are rejected.

For direct execution, use `source.integer_batch_input(carrier, values, grade=1,
limits=limits)` and `PreparedIntegerNetwork.forward_batch`. The result contains
a detached read only buffer and reports workspace capacity and slot count.
Calls on one prepared batch workspace are serialized. Native execution releases
the GIL and checks an owned cancellation token between layers and channels.
This profile provides serial CPU inference; training and rational batch actions
use their separate profiles.

The [batch example](examples/integer_batches.py) checks exact scalar parity and
measures complete binding and Core calls, with source and input setup reported
separately. Its sample columns are field inputs, not additional model parameters.

```sh
python bindings/cpp/examples/integer_batches.py --cells 4096 \
  --batches 1 8 32 --repeats 5 --output timings.json
```

## Exact channel training

The archive must already declare full support gains and a scalar bias on
explicit parameter carriers. Declare their coordinate identifiers and ordered
stored input and target fields. The trainer uses half the metric squared error
on the original channel order. Finite unit and diagonal metrics are supported.

```python
from fractions import Fraction
from rexgraph.cpp_learning import cpp_trainable_model, cpp_training_batch
from rexgraph.model_runtime import train_model

declaration = {
    "grade": 1, "channels": ("identity", "hodge_down", "hodge_up", "hodge"),
    "model": "local-channel-model", "gains": 4000, "bias": 4001,
    "channel_coordinates": 900, "bias_coordinates": 901,
    "history_namespace": 700, "metadata_carrier": 9900, "metadata_tensor": 9901,
    "time_coordinates": 9902, "metadata_coordinates": 9903,
    "initial_parameter_knowledge_time": 2, "publication_cutoff": 0,
    "samples": ({"input": 1001, "target": 1001, "event_time": 3,
                 "knowledge_time": 4, "label_available_time": 5,
                 "partition": "training"},),
}
state = cpp_trainable_model(
    source, (10, 11, 12), declaration,
    metric_keys=(2000, 2001, 2002), metric_selection="diagonal",
)
batch = cpp_training_batch(
    state, source, rate=Fraction(1, 1000000), event_cutoff=3, knowledge_cutoff=5,
)
child = train_model(state, source, batch, steps=4)
output = infer_model(child, source, {"field_key": 1001})
save_model("trained.rcbd", model_record(source, child))

restored, source = read_model_record(load_model("trained.rcbd"))
batch = cpp_training_batch(
    restored, source, rate=Fraction(1, 1000000), event_cutoff=3, knowledge_cutoff=6,
)
next_child = train_model(restored, source, batch)
```

The history namespace must differ from the data namespace. Parameter and
checkpoint identifiers must match the native declaration. Training cycles
through the declared sample order. Evaluation partitions, future labels and
future fitted parameters cannot update the model. The batch retains the exact
declared observations; changing their values, order or mask rejects the update.
Successful updates return an immutable child without changing the parent.
Parameter history is stored separately from the original data publication.

`cpp_training_batch` retains ordered stored field references by default. Inputs
and targets remain in the exact source publication instead of being copied into
Python arrays. The batch's target array has one row per source coordinate and
zero columns; its sample declarations identify the stored target fields.
Batch records use the existing model codecs and require that source when
trained. Use `capture="values"` to retain the materialized input and target
coefficients. Both profiles check the complete sample order, availability,
source identity and observation mask, and produce the same exact updates.
The prepared training example accepts `--capture stored` or `--capture values`
and reports batch capture time separately from update time.

The default executor caches exact channel and target metric pairings for up to
eight samples. Pairings are prepared after the selected sample passes its
availability and partition checks. Every original gain and scalar bias remains
declared. Repeated channels share responses. When both Hodge sectors are
declared, their sum supplies the Hodge response. Gradients return to every
original parameter coordinate. Set `training_cache_samples=0` in `cpp_trainable_model` to use direct
field gradients, or select a cache bound from 1 through 1024. The bound is capped
by the declared sample count and retained in the model configuration.

Direct `source.trainer(..., cache_samples=8)` calls select the same cache.
`NativeTrainer.training_execution` reports physical cache capacity, retained
rational slots, hit/miss counters and the last admitted work and bit envelope.
`response_channels` names the cached responses, with `scalar_bias` last.
Each row in `parameter_coordinates` lists the response indices for one original
gain or bias. These are execution coordinates rather than new learned parameters.
Parent and child handles share this cache under a native lock. Counters remain
outside checkpoint payloads. Restore replays every update and checks complete
canonical history bytes while reusing fixed pairings.
The work envelope covers pairings and parameter arithmetic. Native history
construction and archive validation retain their separate storage limits.

Preparing pairings costs work and storage before reuse; a first update can be
slower than direct field gradients. Selected resource bounds and the native
source work cap still apply. Rational slots do not measure GMP heap bytes.
The [training example](examples/prepared_training.py) compares complete direct
and cached updates and checks an exact synthetic task on held out field samples.

```sh
python bindings/cpp/examples/prepared_training.py --cells 128 \
  --steps 16 --repeats 3 --output training-timings.json
```

The [channel training example](examples/channel_training.py) selects up to 64
original gain coordinates, checks exact replay and reports held out loss and
complete native and Core timings.

```sh
python bindings/cpp/examples/channel_training.py --channels 64 --cells 64 --repeats 3
```

The native contract bounds training to 1024 samples, 64 ordered channels and
1024 updates. Checkpoint metadata uses 4096 bytes by default. Set
`declaration["metadata_capacity"] = 65536` for larger sample declarations,
or select a capacity from 8 through 65536 bytes. The complete declaration and
each update's rate and cutoffs must fit the selected capacity. Smaller capacities
reduce checkpoint correspondence storage when the complete metadata fits.
Capacity stays fixed within a parameter history and survives model records.
Larger capacities retain larger carrier correspondences in each history frame.
Archive byte limits can stop updates before the retained update count limit.
Checkpoint restore validates the complete history by deterministic replay.
Live updates reuse up to eight validated model handles. This is a bounded serial
exact training profile; it does not provide large model training or integer
network backpropagation.

The example also accepts explicit training and held out sample counts:

```sh
python bindings/cpp/examples/prepared_training.py --cells 32 --metric unit \
  --training-samples 256 --heldout-samples 16 --metadata-capacity 16384 \
  --steps 16 --repeats 3 --output sample-training.json
```

## Sparse integer training

`cpp_perceptron_model` uses the native checked integer perceptron. Model rows
use a nonempty source grade carrier. Initial weights require a full support
field on an explicit feature carrier; the bias requires a full support scalar
field. Declare label availability for each source row and the feature and time
coordinate identifiers.

```python
from rexgraph.cpp_perceptron import (
    cpp_perceptron_model, cpp_perceptron_input, cpp_perceptron_batch,
)

availability = tuple({
    "event_time": 1, "knowledge_time": 1, "label_available_time": 1,
    "partition": "training",
} for _ in range(3))
state = cpp_perceptron_model(
    source, 11, 4000, 4001, availability,
    feature_coordinates=900, time_coordinates=999,
)
rows = (((0, 2), (1, 1)), ((0, -2), (1, -1)), ((0, 1), (0, 1), (1, 2)))
batch = cpp_perceptron_batch(
    state, source, rows, (1, -1, 1), rate=2,
    event_cutoff=1, knowledge_cutoff=1,
)
child = train_model(state, source, batch, steps=4)
output = infer_model(child, source, cpp_perceptron_input(child, source, rows))
save_model("integer.rcbd", model_record(source, child))
```

The source in the runnable example supplies these carriers and parameter keys.
Each sparse row contains ordered `(feature_index, integer_value)` occurrences.
Duplicate indices retain their order. Labels are integers `-1` or `1`; the
rate is a positive integer. Prediction maps a zero score to `1`, while training
updates a zero margin. Arithmetic checks every signed 64 bit operation and
rejects overflow without changing the parent model.

`steps` counts complete passes over the observed training rows in primary
order. Use `observed` to mask rows; evaluation rows and unavailable labels cannot
update parameters. Cutoffs must also permit the current fitted parameters.
Checkpoints retain initial parameters, sparse observations, masks, rates and
cutoffs, and validate restored parameters by replay. History is bounded to 1024
epochs and the selected sparse entry limit. Each update copies the parameter
vector. This is a serial learning profile.

Use `parameter_storage="packed"` in `cpp_perceptron_model` to store weights and
bias in one int64 array that cannot be mutated. Initial fields load and validate in C++
without expanding the feature carrier into Python values or support indices.
The default `"values"` storage retains tuples. Both use the same ordered
integer updates, availability checks and checkpoint replay. Packed arrays use
the existing model codecs and application routes. The packed profile still
copies the full parameter vector on each update and retains sparse epoch
observations for replay.

Use `cpp_perceptron_input(state, source, rows, cpu_workers=4, rows_per_tile=256)`
for parallel inference. The optional policy accepts 1 through 64 workers and
1 through 1,000,000 rows per tile. The caller counts as one worker. Each row
retains its feature occurrence order, and output rows retain primary order.
The execution receipt records requested workers, actual worker slots, launched
threads and completed tiles. Counts apply to one call; they do not reserve
cores across concurrent callers or select a count from the host hardware.
The default inference path remains serial. Training retains its serial update
order. Direct binding calls use `NativePerceptron.forward(rows, cpu_workers=4)`.

Use `cpp_perceptron_input(state, source, rows, prepare=True)` to prepare repeated
inference on the same captured input. Core retains up to eight input owners.
Each owner keeps immutable Python rows and an owned native copy. Eviction or
serialization drops the preparation benefit; the portable input record remains
valid and uses ordinary validation. This entry bound is not a heap quota.
Source checks, model seals, resource admission and ordered arithmetic still
run on each inference call. Preparation is independent of parameter values.
Direct binding calls use `prepared = native.prepare_rows(rows)` followed by
`native.forward(prepared)`. The native rows retain their declared feature count
and require the same count on subsequent models. Prepared handles are runtime
objects and do not enter checkpoints. Training uses the existing input profile.

The [CPU inference example](examples/perceptron_cpu.py) measures input capture,
complete binding calls and Core calls with the same sparse observations. It
checks exact scores against an independent integer calculation. The parameters
are declared for inference; the example does not train or evaluate model quality.
Add `--compare-prepared` to measure ordinary and prepared calls, with input
capture and native preparation reported separately.

```sh
python bindings/cpp/examples/perceptron_cpu.py --features 65536 --cells 8192 \
  --workers 1 2 4 8 16 32 --output cpu-inference.json
```

Existing RCQL `MODEL_STATE`, `MODEL_TRAIN`, `MODEL_INFER` and `MODEL_RECORD`
operations accept these saved model records. `MODEL_INIT` declares `cpp_field`,
`cpp_integer`, `cpp_metric` or `cpp_perceptron` models over the native data
attached to a selected model record, using the corresponding factory's keyword arguments as its
configuration. Use a `field_key` input record for
metric inference across restored publications. An Agent callable worker can
invoke `infer_model`. System previews retain exact parameter counts, resume
status and native execution receipts.

## Source and parameter transport

`cpp_transport` declares a structural bijection between two native publications.
Supply a complete cell permutation for each retained grade, explicit carrier
and tensor links, and the destination model factory arguments. Carrier links
declare source and target keys and a coordinate permutation. Source grade
carriers must use the corresponding cell permutation.

```python
from rexgraph.cpp_transport import cpp_transport
from rexgraph.model_runtime import transport_model
from rexgraph.source_binding import CppSourceReference

mapping = cpp_transport(
    source, destination, cell_maps, carrier_links, tensor_links,
    configuration=destination_configuration,
)
child = transport_model(
    state, source, CppSourceReference(destination), mapping,
)
save_model("transported.rcbd", model_record(destination, child))
```

The [transport example](examples/transport_models.py) constructs both sources,
trains an integer model, transports its current parameters, then restores and
continues training.

```sh
python bindings/cpp/examples/transport_models.py /path/to/output
```

This profile supports all four native adapters. It preserves signatures,
ordered occurrences, inactive attachments, semantic scales and lattice scales.
Linked stored fields and metrics must preserve values and structural support.
For trained models, destination parameter fields must already contain the
transported current weights or gains and bias. Declare their coordinate maps;
matching dimensions alone do not establish a map.

The child retains the original model and archive as replay evidence, bounded
to 32 MiB in total. Restore verifies that evidence and the destination
initialization. Training starts a destination history while retaining the
cumulative step count and fitted parameter knowledge time. Rational models
require a new parameter history namespace and rotate the declared samples to
the original next sample. Time coordinate identifiers and times remain shared
by explicit declaration. Optimizer transport accepts only `optimizer="reset"`.

Each lineage permits one transport. Complete bijections are required;
rectangular maps, chained transports and source evolution are unsupported.
The existing cumulative limit of 1024 training updates or epochs still applies.
RCQL `MODEL_TRANSPORT` accepts the same mapping and a destination native model
record binding. System previews show the origin model and destination history
step without expanding the retained proof.

For direct field transport, `source.transport(destination, cell_maps,
carrier_links)` returns an owning handle. `push_values(carrier_key, values)`
returns exact destination values and their carrier. `check_tensor(source_key,
target_key)` checks linked stored field or metric values and support. Declare
resource limits when constructing the handle and pass an owned cancellation
token to each call. Transport uses synchronous CPU execution.

## Persistence and execution

Model records preserve canonical native archive bytes, source identity,
temporal history, selected metrics and model declarations. `.rcbd`,
`.safetensors` and existing RCDB stores use the same model record route.
Restore checks the compiled engine and bridge receipts before returning a
prepared model. Prepared handles and caches are rebuilt from declarations.

`source.at(sequence)` returns an owning prefix without later frames. Each
publication has its own live identity. A prepared action rejects fields from
a separately loaded publication, even when their archive bytes agree.

Archives have a default byte limit of 256 MiB. `cpp_model` accepts five ordered
resource limits: binding records, reference rational slots, reference sparse
entries, scalar work and integer bits. These bounds apply to the native
reference containers rather than total process memory. `source.prepare` reuses up to eight selected
plans. Native preparation and execution release the Python GIL. Calls on the
same prepared plan are serialized.

For direct execution, pass a `CancellationToken` to `PreparedField.execute`.
Cancellation is checked at native phase boundaries. `ExecutionLimitError`
exposes the exceeded resource, required bound and selected limit.
`PreparedIntegerNetwork.forward`, `NativeTrainer.forward` and
`NativeTrainer.advance` also accept an owned cancellation token.
`NativePerceptron.forward` and `NativePerceptron.advance` use the same cancellation
contract. The perceptron checks cancellation between inference rows and at
training epoch boundaries.

## Graph history

`NativeAlgebraicGraphRevision` selects a declared graph channel at one native
publication. `NativeAlgebraicGraphDatabase` returns stable archive publications
on that revision's actual owners. Archive reads require an availability cutoff.

```python
from rexgraph_cpp import NativeAlgebraicGraphRevision, NativeAlgebraicGraphDatabase
from rexgraph.cpp_algebraic_query import algebraic_snapshot_record, read_algebraic_snapshot_record

revision = NativeAlgebraicGraphRevision(source, (9611, 1, 2), sequence=41)
availability = (revision.manifest["namespace"], 41)
database = NativeAlgebraicGraphDatabase(revision, availability=availability)
snapshot = database.archive(1, availability=availability)
record = algebraic_snapshot_record(snapshot)
restored = read_algebraic_snapshot_record(record)
```

The record retains the complete native prefix, anchored RFCXH declaration,
source bindings and RFCXG graph. Use the existing query, coefficient window,
RCBD, safetensors and RCDB interfaces on the selected publication.
`ALGEBRAIC_AT(sequence)` and `ALGEBRAIC_HISTORY_INFO()` also accept these records.
Separate restorations create independent owners. Graph history origin comes
from database selection. Standalone publications retain their supplied labels.

`NativeAlgebraicGraphRevision.default_limits()` lists the separate graph,
snapshot, binding and revision limits. Saved limits and reader overrides both
apply on restore. Selecting an earlier endpoint excludes later graph payloads;
selection validates the graph at that endpoint.

## Tests

Install Core, RCDB, the bindings and pytest. The tests compile an SDK consumer
that generates native archives and canonical exact reference values.

```sh
CMAKE_PREFIX_PATH=/path/to/installed/sdk python -m pytest bindings/cpp/tests -q
```

Phase/fold cases use `tests/storage/fixtures/phase_fold_v1` from the C++ source
as `REXGRAPH_CPP_PHASE_FOLD_REFERENCE`. Run the `phase_fold_fixture` CMake
target's executable with that directory and an output directory, then set
`REXGRAPH_CPP_PHASE_FOLD_FIXTURES` to its output when running pytest.

Graph history cases use the C++ `graph_database --export` output as input to
`algebraic_graph_history_fixture INPUT_DIRECTORY OUTPUT_DIRECTORY`. Build that
fixture with `REXGRAPH_BINDING_TEST_FIXTURE=ON`, then set
`REXGRAPH_CPP_GRAPH_HISTORY_FIXTURES` to its output when running pytest.

## Whole grade algebraic fields

`NativeAlgebraicGradeTensor` selects one vector record for each original cell
of a grade. The record order is the original cell order. Each vector must use
that cell's actual declared stalk. `stalk_offsets` retains heterogeneous ranks;
`source_coordinate(i)` returns the original grade, cell index and local stalk
coordinate for a flattened coefficient.

```python
from rexgraph_cpp import NativeAlgebraicGradeTensor, NativeAlgebraicGradeTensorWindow
from rexgraph.cpp_algebraic_grade import algebraic_grade_record, read_algebraic_grade_record
from rexgraph.io import save, load

availability = (snapshot.namespace, snapshot.sequence)
field = NativeAlgebraicGradeTensor(snapshot, grade, record_keys, availability=availability)
lower_field = field.boundary(availability=availability)
size = lower_field.domain[0]
window = NativeAlgebraicGradeTensorWindow(lower_field, [size], [0], [size],
                                         availability=availability)
save("grade.rcbd", algebraic_grade_record(window, availability=availability))
restored = read_algebraic_grade_record(load("grade.rcbd"))
```

Boundaries require a complete declared fiber system, including for zero inputs.
A derived field retains its input owner and source publication. Windows require
a shape covering the full grade, then select a coordinate box. Empty windows
have no sum; supported zero coordinates retain complete polynomial tuples.
`sum` contracts coefficient coordinates and does not declare a metric or a
gluing map. Execution limits remain bounded by the retained input policy.

`algebraic_grade_record` supports RCBD and safetensors records. Native restoration
replays the selected boundary lineage and checks exact coordinates against the
saved observation. `read_algebraic_grade_observation_record` checks portable
observations and retained source bytes without loading the bindings.

RCQL exposes `ALGEBRAIC_GRADE(request, max_sequence)` and
`ALGEBRAIC_GRADE_WINDOW(request, shape, lower, upper, max_sequence)`. The request
contains `grade`, `record_keys` and `boundary_steps`. Both operate on a selected
algebraic snapshot record under its read policy and explicit temporal cutoff.

## Whole grade model actions

`cpp_algebraic_grade_model` declares a fixed exact boundary action on an owning
`NativeAlgebraicSnapshot`. Its input keys select one vector for each original
cell. `boundary_steps=0` returns the selected field; larger values follow the
native boundary lineage. Input and output coordinates name original cells and
local stalk coordinates, with a separate polynomial coefficient axis.

```python
from rexgraph.cpp_algebraic_grade_model import cpp_algebraic_grade_model
from rexgraph.model_runtime import infer_model
from rexgraph.io.model_record import model_record, save_model, load_model, read_model_record

state = cpp_algebraic_grade_model(snapshot, grade=grade, record_keys=record_keys,
                                 boundary_steps=1, availability=availability)
output = infer_model(state, snapshot)
save_model("model.rcbd", model_record(snapshot, state))
restored_state, restored_snapshot = read_model_record(load_model("model.rcbd"))
restored_output = infer_model(restored_state, restored_snapshot)
```

An input can be an owning whole grade field on the same actual publication, or
a complete output from another whole grade model. Portable inputs replay their
retained record selection and boundary steps before evaluation. Equal archive
bytes in independently restored publications do not identify live input owners.
`infer_algebraic_grade_model` additionally accepts explicit availability,
execution limits and a native cancellation token. Reader limits can narrow saved
archive admission. They do not alter the source's original maps or coordinates.

The selected source owns the action's fixed fiber maps. This profile exposes
construction, preparation and inference; it declares no training or parameter
transport. Algebraic outputs keep every `Fraction` coefficient. Numerical input
conversion requires an explicitly selected field embedding.

RCQL uses `MODEL_INIT("cpp_algebraic_grade", configuration)` on an algebraic
snapshot record, followed by the existing `MODEL_INFER`, `MODEL_RECORD`,
`MODEL_VALUES` and `MODEL_INFO` operators. Configuration contains `grade`,
`record_keys`, `boundary_steps` and `availability`, with optional execution and
coordinate limits. Agent workers and System previews use the same model outputs.
The [whole grade model example](examples/algebraic_grade_models.py) selects a
graph publication, saves a model record and checks restored exact inference.

## Original stalk model composition

`prepare_algebraic_grade_stalks(output, snapshot)` replays a whole grade model
output once and retains its actual owning result. `select(cell)` returns an exact
vector output on that original cell's carrier. `native(cell)` returns its owning
native vector. `select_many(cells, max_selections=256)` preserves the requested
order and repeated cells. A supplied `tensor` must match the actual publication,
complete coordinates and retained boundary lineage. Empty grades have no stalk.

```python
from rexgraph.cpp_algebraic_stalk import prepare_algebraic_grade_stalks
from rexgraph.cpp_algebraic import cpp_algebraic_model, cpp_algebraic_input

selection = prepare_algebraic_grade_stalks(grade_output, snapshot)
vector = selection.select(cell)
archive = snapshot.archive_owner
state = cpp_algebraic_model(archive, operator_key=operator_key,
                            input_key=carrier_record, operation="solve")
inputs = cpp_algebraic_input(state, archive, vector)
output = infer_model(state, archive, inputs)
```

Carrier models require the selected original carrier, including its grade and
cell identity. Equal ranks do not imply compatibility. Map, metric solve,
quadrance, covector and evaluation actions use their existing native operators.
Captured selected inputs and subsequent model outputs retain the complete grade
origin through the model codecs. This input profile admits at most 32 composed
actions and bounds packed origin bytes by the model's archive reader limit.

`algebraic_grade_stalk_record` retains the selected output, complete source
publication and native prefix in RCBD or safetensors. The portable
`read_algebraic_grade_stalk_output` checks the output and retained bytes without
native bindings. `read_algebraic_grade_stalk_record` restores the publication,
replays every boundary and returns `(output, snapshot, native_vector)`.

RCQL `MODEL_STALK(output, cell)` selects a stalk under the selected source's read
policy. System previews show the original cell and source coordinate slice.
The [stalk model example](examples/algebraic_stalk_models.py) loads a whole grade
model record, applies a carrier model and checks restored exact inference.

### Whole grade metric actions

Declare one certified metric record for every retained cell, including cells on
other grades. The records must belong to the selected publication and its metric
domain. A prepared field retains the occurrence adjoints for repeated actions.

```python
from rexgraph_cpp import NativeAlgebraicMetricField
from rexgraph.cpp_algebraic_metric import algebraic_grade_metric_record

metrics = NativeAlgebraicMetricField(snapshot, metric_keys,
    availability=(snapshot.namespace, cutoff), max_retained_coordinates=1000000)
hodge = metrics.apply(tensor, "hodge", availability=(snapshot.namespace, cutoff))
up = metrics.apply(tensor, "coboundary", availability=(snapshot.namespace, cutoff))
energy = metrics.adjacent_quadrance(tensor, availability=(snapshot.namespace, cutoff))
pairing = metrics.inner(tensor, other, availability=(snapshot.namespace, cutoff))
record = algebraic_grade_metric_record(metrics, tensor, "hodge",
    availability=(snapshot.namespace, cutoff))
```

The coboundary target is the next original grade. Hodge preserves the input
grade and includes both adjacent sectors when present. Results retain complete
polynomial coefficients. The first and last grades have exact zero contributions
from their absent sectors. Vector results expose original stalk offsets and
`stalk(cell)` on the actual result carrier.

RCBD and safetensors retain the complete source publication, selected metrics and
input boundary lineage. `read_algebraic_grade_metric_observation_record` reads
portable declarations without native imports. `read_algebraic_grade_metric_record`
restores the source and checks results by native replay. Reader metric admission
can narrow the saved retained coordinate limit. Cancellation is checked between
native phases. These actions are fixed operators; they do not declare a trainer.

RCQL exposes `ALGEBRAIC_GRADE_METRIC(request, metric_keys, operation, max_sequence)`
through the existing source read policy. The request names the original grade,
vector keys and boundary steps. The optional `other_request` supplies the second
input for `inner`. `examples/algebraic_grade_metrics.py` runs and saves one action.

### Whole grade metric models

`cpp_algebraic_grade_metric_model` declares a saved fixed metric action using the
same complete publication and explicit metric keys.

```python
from rexgraph.cpp_algebraic_grade_metric_model import cpp_algebraic_grade_metric_model
from rexgraph.io.model_record import model_record, save_model, load_model, read_model_record
from rexgraph.model_runtime import infer_model

request = {"grade": grade, "record_keys": vector_keys, "boundary_steps": 1}
state = cpp_algebraic_grade_metric_model(snapshot, request=request,
    metric_keys=metric_keys, operation="hodge", availability=(snapshot.namespace, cutoff))
output = infer_model(state, snapshot)
record = model_record(snapshot, state)
save_model("metric.rcbd", record)
restored_state, restored_snapshot = read_model_record(load_model("metric.rcbd"))
restored_output = infer_model(restored_state, restored_snapshot)
```

The request's boundary steps prepare the default input. Supplied inputs already
occupy the resulting original grade; those steps are not applied again. Inputs
can be owning grade tensors or complete boundary model outputs on the same
publication. Boundary model outputs retain their complete producer receipt in
the metric output. `inner` requires an explicit `other_request`; supplied operands
use `{"input": first, "other": second}`. Hodge and coboundary return original
grade coordinates. Inner and adjacent quadrance return one polynomial scalar.

Preparation retains at most eight native field and input plans. Execution limits
bound input replay, and `max_retained_coordinates` bounds metric preparation.
Inference overrides can narrow those saved limits. RCQL uses
`MODEL_INIT("cpp_algebraic_grade_metric", configuration)` with the same options,
then `MODEL_INFER`, `MODEL_RECORD`, `MODEL_VALUES` and `MODEL_INFO`. Model records
support RCBD and safetensors. These fixed models expose no training or parameter
transport. The [metric model example](examples/algebraic_grade_metric_models.py)
checks exact inference after saving and restoring a graph publication.

### Metric result stalk composition

`prepare_algebraic_grade_stalks` also accepts complete Hodge and coboundary model
outputs. It returns original cell vectors through the same `select`, `native` and
`select_many` methods. Pairing and quadrance scalars have no cell stalks.

```python
selection = prepare_algebraic_grade_stalks(metric_output, snapshot)
vector = selection.select(cell)
archive = snapshot.archive_owner
carrier_model = cpp_algebraic_model(archive, operator_key=cell_metric_key,
                                   input_key=cell_vector_key, operation="solve")
inputs = cpp_algebraic_input(carrier_model, archive, vector)
output = infer_model(carrier_model, archive, inputs)
```

Selected vectors retain the complete metric model output, including selected
metrics and every input model origin. Subsequent carrier model actions retain
that selection through the existing bounded origin profile. Carrier grade and
cell identity must match; equal ranks do not establish compatibility.

Preparation replays the metric action once. The optional `metric_values` argument
can reuse actual owning native metric results after checking the publication,
operation, metrics, original input lineage and complete result coordinates.
`max_retained_coordinates` and `max_origin_bytes` can narrow the saved metric and
producer byte admission. Reusing supplied results requires their retained metric
limit to satisfy the requested bound. Input replay uses the saved execution
policy, and cancellation remains checked between native phases.

The existing `algebraic_grade_stalk_record`, `read_algebraic_grade_stalk_output`
and `read_algebraic_grade_stalk_record` functions accept these selections in RCBD
and safetensors. Portable reads require no native bindings. Native reads replay
the full metric action and return the vector on its actual result carrier.
RCQL `MODEL_STALK(output, cell)` and System previews use the same original cells.
Carrier inference accepts complete stalk outputs directly. RCQL can compose
`MODEL_INFER(carrier_model, MODEL_STALK(metric_output, cell))` through the same
checked origin capture. Subsequent vector outputs retain their bounded origin
chain when supplied directly to another carrier model.
The [metric stalk example](examples/algebraic_metric_stalk_models.py) composes a
saved metric model output with a carrier model and checks restored exact results.

### Coefficient features for integer learning

`cpp_coefficient_perceptron_input` binds complete algebraic vector, covector or
scalar outputs to an existing `cpp_perceptron` model. Supply the learner's ordered
`row_keys` and one feature declaration per row. Each feature names a parameter
index, module coordinate, polynomial power and exact scale:

```python
inputs = cpp_coefficient_perceptron_input(
    classifier, learner_source, vectors, row_keys=classifier.space.keys,
    features=(((0, 0, 0, 2), (1, 0, 1, 3)),) * len(vectors), prepare=True)
prediction = infer_model(classifier, learner_source, inputs)
```

The declared scale must produce a signed 64 bit integer. Floating scales,
rounding and repeated parameter indices are rejected. Explicit zero features
retain their support and order. These are observations of polynomial coordinates;
the original algebraic fields and carriers stay in the retained producer outputs.
Learner row links and feature indices are declared independently of producer
cell positions. Equal dimensions do not create a mathematical map between them.

`cpp_coefficient_perceptron_batch` accepts the same declarations plus signed
labels, a positive integer rate, and event and knowledge cutoffs. Existing label
availability and training partition checks apply. Cutoffs use the learner's time
coordinates; producer publications retain their own namespaces and cutoffs.
Training history and inference receipts retain complete producer outputs once
per input declaration, including earlier metric and boundary model origins.
`max_origin_bytes` bounds a complete feature declaration, default 16 MiB and at
most 256 MiB. Complete training history has a separate 256 MiB feature byte bound.
Prepared inputs use the existing owning sparse rows and optional CPU tile policy.

Model records, RCQL `MODEL_INFER` and `MODEL_TRAIN`, Agent workers and System
previews use the existing integer model routes. RCBD and safetensors retain the
feature observations. Portable record decoding checks the coefficient links
without native bindings. Retain the original algebraic records separately when
the original producer action needs native replay.
The [coefficient feature example](examples/coefficient_feature_models.py) reads
an existing learner record and a JSON array of rows with `row`, `record`,
`features` and, for training, `label`. Each `record` names a saved stalk record;
exact scales use integer or fraction strings. It trains under explicit cutoffs
and checks inference after saving and restoring the classifier.

### Packed Boolean gate fitting

`cpp_binary_gate_model` declares two named Boolean input channels on an original
source grade. Its parameter field has full support on four coordinates, ordered
as `(0, 0)`, `(0, 1)`, `(1, 0)` and `(1, 1)`. Supply the parameter and time
coordinate namespaces and one availability declaration per source row.

```python
state = cpp_binary_gate_model(
    source, carrier, parameter_key, availability, grade=0,
    parameter_coordinates=parameter_coordinates, time_coordinates=time_coordinates)
first = pack_binary_plane(first_values)
second = pack_binary_plane(second_values)
batch = cpp_binary_gate_batch(
    state, source, first, second, pack_binary_plane(labels),
    event_cutoff=event_cutoff, knowledge_cutoff=knowledge_cutoff)
state = train_model(state, source, batch)
inputs = cpp_binary_gate_input(state, source, first, second, prepare=True)
output = infer_model(state, source, inputs)
```

Planes contain exact zero or one coefficients, with lane zero in the least
significant bit. Packed inputs require zero tail padding. Physical packed lanes
retain the declared source row order. Fitting checks all 16 truth tables and
minimizes the integer Hamming count on observed training rows. Ties select the
lowest truth table. Unobserved input states therefore receive zero. The source
complex remains fixed. Repeating a fit on the same observations retains the
same optimal gate and records each admitted fit in the model history.

The default lane bound is 1,048,576, with a maximum of 64,000,000. Complete fit
history has a separate byte bound, default 16 MiB and maximum 256 MiB, and at
most 1,024 entries. Native cancellation is checked before and after each phase.
Output values use one byte per source row and receipts retain the packed plane.
Parameter knowledge times retain their declared coordinate namespace.
RCBD and safetensors preserve the observations and parameters. Portable decoding
requires no native bindings; native restoration replays fitting. RCQL
`MODEL_INIT("cpp_binary_gate", config)`, `MODEL_INFER` and `MODEL_TRAIN` use the
same model routes as Agent workers, RCDB records and System previews.

The [Boolean gate example](examples/binary_gate_models.py) reads an existing
model record and JSON arrays named `first`, `second` and optional `targets`.
Fitting requires explicit event and knowledge cutoffs. It saves the resulting
model and checks restored inference.

## Named relation stalk fields

`PreparedStalkFields` owns a grade 1 source, the named axes of every original
cell and one exact block map per original occurrence. Upper and lower stalks
may have different dimensions. Supply each block as `(local_row, local_column,
coefficient)` entries. The source supplies the signed incidence coefficient and
activity of that occurrence. Repeated, inactive and supported zero occurrences
retain separate output vectors.

```python
from rexgraph_cpp import PreparedStalkFields

plan = PreparedStalkFields(source.archive, lower_axes, upper_axes, flag_maps)
result = plan.apply((whole_field,))
response = result["fields"][0]
boundary = result["boundaries"][0]
occurrences = result["occurrence_fields"][0]
```

`whole_field` supplies every named coordinate of each original upper stalk in
source order. Each result retains the same ragged stalks. This profile uses
unit stalk metrics and solves `(I + parameter * B.T * B) g = j` exactly through
a prepared augmented system and native boundary actions. It requires a positive
integer or `Fraction` parameter and rejects nonempty higher grades. Limits bound
dimensions, logical retained entries, work and integer bits. Exact factorization
can introduce fill and larger rational coefficients.

`cpp_stalk_fields_model` exposes this action through Core's model interface.
`stalk_annotation_batch` acquires complete stalk vectors on declared training
rows. Each observation retains its own native relation column. Inference injects
their native boundary through a declared coordinate map and returns the complete
response, boundary and occurrence fields. `stalk_query` supplies an additional
whole field; `include_acquisition=False` executes that field alone.

```python
from rexgraph.cpp_stalk_fields import (
    cpp_stalk_fields_model, stalk_annotation_batch, stalk_query, stalk_output_field,
)
from rexgraph.model_runtime import train_model, infer_model
from rexgraph.io.model_record import model_record, save_model, load_model, read_model_record

state = cpp_stalk_fields_model(
    source, lower_axes, upper_axes, flag_maps, training_rows=(training_row,))
state = train_model(state, source,
    stalk_annotation_batch(state, ((training_row, complete_stalk_vector),)))
output = infer_model(state, source, stalk_query(state))
field = stalk_output_field(output, training_row)
save_model("stalk-model.rcbd", model_record(source, state))
state, source = read_model_record(load_model("stalk-model.rcbd"))
```

The model output's rectangular storage envelope has an explicit support mask.
Absent padding has no branch meaning. For different local axes, select one
original row with `stalk_output_field`; common named axes permit `output.tensor()`.
Model records support RCBD, safetensors, RCDB,
RCQL `MODEL_INIT("cpp_stalk_fields", config)`, Agent workers and System previews.
Acquisition retains supplied observations; it does not fit the declared flag
maps or select discrete interpretations from linear field combinations.

`source_accession` also accepts `original-structural-language-source-v1`
declarations. Preparation verifies the original columns, UTF8 byte fields and
complete token maps. Saved models retain these accessions for replay.

### Fields for compatible interpretations

`cpp_stalk_sections_model` connects a joint annotation checkpoint to a named
stalk checkpoint on the same original source. Declare a port for every context
and complete annotation tuple. Each port supplies that context's whole local
vector. Joint adapter versions 1, 2 and 3 are supported.

```python
from rexgraph.cpp_stalk_sections import (
    cpp_stalk_sections_model, stalk_section_annotation_batch,
    stalk_section_query, stalk_section_field,
)

state = cpp_stalk_sections_model(source, sections, fields, ports)
state = train_model(state, source, stalk_section_annotation_batch(
    state, ((training_row, complete_annotation),)))
query = stalk_section_query(state, query_rows)
output = infer_model(state, source, query)
assignments = output.execution_receipt["assignments"]
routes = output.execution_receipt["branch_assignments"]
field = stalk_section_field(output, case_index, query_row)
save_model("interpretations.rcbd", model_record(source, state))
```

Each compatible section retains its own complete response and original
dictionary route witnesses. Identical fields share execution while preserving
separate cases. Selection requires an explicit case index. Optional `route_ports`
supply `(context_row, annotation, native_branch_identity, whole_vector)` entries
for particular compatible dictionary routes.

Optional `interfaces` declare `(name, axes, restrictions)` entries. Each
restriction is `(original_row, sparse_block)`; block entries are
`(interface_row, local_column, coefficient)`. Receipts retain complete interface
readings and vector residuals for both input and response. These readings report
compatibility; they do not remove sections or enforce a field gluing constraint.
Acquisition updates finite annotations and complete vector columns together.
Ports and block maps remain declared. `max_case_entries` bounds the logical
field envelope for all retained cases. Saved models use the existing model
formats and stack routes, including RCQL `MODEL_INIT("cpp_stalk_sections", config)`.

## Joint relational annotations

`cpp_relational_sections_model` acquires complete annotation tuples on retained
context relations. Supply the original occurrence axes and literal accessions
alongside the source. Layer values are named symbols; their indices carry no
semantic distance.

```python
from rexgraph.cpp_relational_sections import (
    cpp_relational_sections_model, joint_annotation_batch, joint_query,
)
from rexgraph.model_runtime import train_model, infer_model

state = cpp_relational_sections_model(
    source, lower_axes, occurrence_axes,
    layers=("meaning", "diction"),
    joint_values=(("sense-a", "literal"), ("sense-b", "idiomatic")),
    training_rows=(training_row,), source_accession=source_accession,
)
batch = joint_annotation_batch(state, ((training_row, ("sense-a", "literal")),))
state = train_model(state, source, batch)
output = infer_model(state, source, joint_query(state, (query_row,)))
assignments = output.execution_receipt["assignments"]
best = output.execution_receipt["best_indices"]
```

Acquired tuples retain native correspondence columns. `compatibility_types`
declares ordered layer indices; annotation `groups` acquire their allowed tuples.
A query chart with `name`, `support` and `type` applies that acquired table to
explicit query uses. Support entries are `(query_index, layer_index)` pairs.
Charts with `allowed` instead of `type` supply an explicit compatibility table.
Contradictory requests return no assignments and retain an exact conflict core.
Section admissions bound finite domains, entries and a logical work envelope.
Scores remain prospective field couplings. RCBD, safetensors and the existing
RCQL, RCDB, Agent and System model routes retain the same declarations.

For context conditions, supply `guard_limits=(4096, 1000000, 64000000, 16384)`
when creating the model. This selects adapter version 2. The bounds cover
candidate bindings, native entries, aggregate native work and integer bits.
Declare the anchored roles for each distinct context in a training group:

```python
from rexgraph.cpp_context_guards import guarded_annotation_group

group = guarded_annotation_group(
    state, "context-meaning", ((training_row, 0),),
    anchors=(("target:0", "left:1"),),
)
batch = joint_annotation_batch(
    state, ((training_row, ("sense-a", "literal")),), groups=(group,),
)
state = train_model(state, source, batch)
output = infer_model(state, source, joint_query(state, (query_row,)))
```

The model must declare `compatibility_types={"context-meaning": (0,)}`. Anchors
include every target role. Other participants may change only through a complete
native ordered bijection of the context and its recorded phrase boundaries.
Shared identities remain shared across all contexts in the group. Matching rules
apply automatically to queried uses and retain their native map witnesses.
Unmatched rules add no constraints. An explicit typed chart requires a matching
guard. Use `discover_guarded=False` to compare inference without automatic rules.
These finite conditions do not infer unrestricted linguistic syntax.

Pass `dictionary=True` to `guarded_annotation_group` to require dictionary
support for substitutions. Each changed literal word must have a retained
record binding. Each changed record pair must share an actual synset through
the original record, entry and sense columns. The response retains every
supporting route and its complete native map. Unknown or unrelated substitutions
receive no rule. A common dictionary sense does not determine the sense used
in a sentence; the response keeps this distinction explicit.

Use `joint_dictionary=True` instead to require one native map over the whole
training group's context, phrase and dictionary routes. Shared entries, senses,
synsets and original columns must agree in that map, including unchanged records.
Every compatible dictionary branch is retained in the response. The candidate
bound covers context bindings and full dictionary choice products before
enumeration; exceeding it rejects the request.

## Graded stalk boundaries

`cpp_stalk_boundary_model` executes downward boundary maps on complete named
stalk fields. Declare `stalk_axes` for every original grade and `flag_maps` for
every original occurrence. Each sparse block contains `(row, column, coefficient)`
entries. Grade zero has an empty flag family. Maps are multiplied by the signed
active source incidence coefficients. All adjacent boundary compositions must
vanish on the complete declared local spaces before the model is prepared.

```python
from rexgraph.cpp_stalk_boundary import (
    cpp_stalk_boundary_model, stalk_boundary_annotation_batch,
    stalk_boundary_query, stalk_boundary_stage,
)
from rexgraph.model_runtime import train_model, infer_model
from rexgraph.io.model_record import model_record, save_model

state = cpp_stalk_boundary_model(
    source, stalk_axes, flag_maps, training_rows=(training_row,),
    grade=2, boundary_steps=2,
)
batch = stalk_boundary_annotation_batch(state, ((training_row, whole_vector),))
state = train_model(state, source, batch)
output = infer_model(state, source, stalk_boundary_query(state))
trace = output.execution_receipt["trace"]
relation_field = stalk_boundary_stage(output, stage=1, row=relation_row)
save_model("graded-fields.rcbd", model_record(source, state))
```

A trace retains complete fields at each requested grade, original occurrence
vectors and intermediate cell channels for each pair of boundary steps.
Nonzero channels remain visible when their sum is zero. Repeated, inactive and
supported zero occurrences keep their original slots and keys. Names and local
dimensions may differ between cells. Select an original row when exposing a
ragged field as a `TensorField`. Acquisition keeps one native column per complete
input stalk observation; repeated observations remain distinct columns.

Use `grade=0, boundary_steps=0` for object fields. This profile declares downward
maps without choosing a metric, upward map or Green operator. It does not infer
higher relations from field scores or fit flag maps. RCQL accepts
`MODEL_INIT("cpp_stalk_boundary", config)`. Existing RCBD, safetensors, RCDB,
Agent and System model routes retain the complete trace. Action bounds cover
local dimensions, logical retained entries, checked work and integer bits.
`replay_stalk_boundary_output(output, source)` verifies the saved trace against
the retained native implementation and its full chain check.

## Local tensor sections

`cpp_local_sections_model` acquires local tensor charts without declaring every
complete annotation tuple. Each stalk names an original lower cell, relation or
occurrence, its local axes and its whole finite tensor alternatives. An observation
names a constraint family, complete source relation rows, stalk support, observed
alternative names and any fixed lower identities.

```python
from rexgraph.cpp_local_sections import (
    enable_local_sections_provider, cpp_local_sections_model,
    local_section_batch, local_section_query, local_section_fields,
)
from rexgraph.model_runtime import train_model, infer_model

enable_local_sections_provider()
state = cpp_local_sections_model(source, stalks, families, training_rows)
state = train_model(state, source, local_section_batch(state, observations))
query = local_section_query(state, variables, target_relation_patches)
output = infer_model(state, source, query)
fields = local_section_fields(output, case=0)
```

Transfer requires a native ordered map over the complete declared source rows.
Types, roles, coefficients, inactive occurrences and shared identities must agree.
Fixed identities remain fixed. Charts compose on shared stalks; joint observations
keep their dependencies. Uncovered stalks retain all declared alternatives with
`supported=False`. Alternatives in one constraint family are disjunctive; different
families apply together. No implicit independence or scalar ranking is inferred.

For relation stalks, `field_model` may declare a `cpp_stalk_fields_model` with the
same source and local bases. `field_constraints` admit cases using whole native
Green response vectors. `response-boundary-zero` checks original lower stalks;
`response-occurrence-equality` compares original occurrences on one shared lower
stalk. These are downward field conditions, separate from observation sheaf maps.
Receipts retain every finite case, field output and admission check.

RCQL accepts `MODEL_INIT("cpp_local_sections", config)`. Existing model storage
and worker routes retain the observations and tensors. Select a case explicitly
when reading ragged fields. `replay_local_sections_output` checks the native maps,
all sections and declared field constraints. This profile requires supplied
structural observations; it does not parse grammar or infer intent from raw text.

### Exact incidence transport learning

`cpp_stalk_transport_model` learns declared map entries from complete field
observations on the original grade 1 source. `stalk_axes` names every original
stalk. `fixed_maps` declares complete original flag blocks; `parameter_slots`
lists distinct `(occurrence, row, column)` entries disjoint from fixed entries.
`training_rows` selects the lower stalks that may be observed. Higher grades
must be empty. An optional `source_accession` retains checked linguistic and
literal byte maps.

```python
from rexgraph.cpp_stalk_transport import (
    cpp_stalk_transport_model, stalk_transport_observation_batch,
    stalk_transport_query, stalk_transport_field, stalk_transport_maps,
)
from rexgraph.model_runtime import train_model, infer_model

state = cpp_stalk_transport_model(
    source, stalk_axes, fixed_maps, parameter_slots, training_rows)
batch = stalk_transport_observation_batch(state, input_field, annotations)
state = train_model(state, source, batch)
output = infer_model(state, source, stalk_transport_query(state, query_field))
```

Annotations are `(lower_row, complete_vector)` pairs. Input and query fields
cover every named upper stalk. Output retains an `origin` component and each
nonzero query direction on the `affine-component` axis. Its receipt keeps
boundary and occurrence families separately, with exact identifiability masks.
Directions whose complete boundary and occurrence responses vanish for the
query are recorded in `zero_components`. The complete map kernel remains sparse
in the model checkpoint, with its declared shape and original free coordinates.
`stalk_transport_field(output, row)` requires an identified requested stalk.
`stalk_transport_maps(state, source)` requires a uniquely identified map;
otherwise pass explicit `free_parameters`. Materialized `flag_maps` can be used
by the existing boundary and Green models. Inactive and supported zero flags
retain their underlying unidentified map directions.

RCQL accepts `MODEL_INIT("cpp_stalk_transport", config)`. Existing model records,
stores, prediction workers and System previews retain the observations and
affine families. Restore replays the original batches and complete native fit.

Several original batches can be fitted together with one child checkpoint:

```python
from rexgraph.cpp_stalk_transport import stalk_transport_observation_group

group = stalk_transport_observation_group(state, batches)
state = train_model(state, source, group, steps=len(batches))
```

`batches` is an ordered sequence of original observation batches. The group
retains every original input field, target, mask and batch checkpoint. `steps`
must equal the number of original observations. The complete parameter family
is fitted jointly and the child refers to the supplied parent state. Observation
counts and retained checkpoint entries must fit the model's declared limits.
Groups use the existing model codec, storage formats and `MODEL_TRAIN` operator.

For encoded sparse fields on original columns, select `column_fields=True` when
creating the model. Use `column_transport_observation` and
`column_transport_observation_group` from `rexgraph.cpp_column_transport`:

```python
from rexgraph.cpp_column_transport import (
    column_transport_observation, column_transport_observation_group,
)

state = cpp_stalk_transport_model(
    source, stalk_axes, fixed_maps, parameter_slots, training_rows,
    column_fields=True,
)
observation = column_transport_observation(
    state, {0: ((0, 1),)}, {0: ((0, 1),)},
)
group = column_transport_observation_group(state, (observation,))
state = train_model(state, source, group, steps=1)
```

Input keys name original upper cells. Target keys name observed lower cells.
Each value supplies ordered nonzero coordinates in that cell's complete stalk;
an empty tuple declares a zero stalk. Omitted input columns have the declared
zero field. Omitted target stalks are unobserved. This version uses independent
original map parameters and keeps one checkpoint per ordered batch. Context
transfer currently materializes whole fields within the model's action limits.

Training reuses the prepared native source and inserts only newly acquired
observation equations. `PreparedStalkTransport.append(observations)` returns an
immutable native child for a nonempty original observation suffix. Its
`execution_stats` reports source reuse, reused and assembled observation counts,
and guarded work. The fit receipt retains the canonical full replay cost in
`preparation_steps`; cold restore checks the complete family and receipt.

For shared map parameters, also pass `parameter_space` and `parameter_basis`.
The space is a `CoordinateSpace` or a mapping with `name` and ordered `keys`.
Each nonzero basis entry is `(entry_index, parameter_index, exact_coefficient)`;
`entry_index` selects an original coordinate from `parameter_slots`. Multiple
parameters may contribute to one entry, and one parameter may control entries
on several original flags. These declarations select adapter version 2.

```python
from rexgraph.type_accession import CoordinateSpace

state = cpp_stalk_transport_model(
    source, stalk_axes, fixed_maps, parameter_slots, training_rows,
    parameter_space=CoordinateSpace("shared-gains", ("meaning", "grammar")),
    parameter_basis=parameter_basis)
```

Responses name free directions by parameter key. `MODEL_INFO` and System previews
report the named parameter count, supported map entry count and basis rank.
Materialization requires values for every free named parameter, including aliases
between dependent templates. Supported entries without a basis term remain zero.

### Conditional field objectives

`cpp_stalk_objective_model` applies a declared field action within an observed
transport family. Each named view lists `(occurrence, lower_axis, coefficient)`
terms. A term contributes to its original upper relation. Probes contain complete
upper stalk fields. The scalar relation metric is unit, with distinct participants
per relation as required by the channel profile.

```python
from rexgraph.cpp_stalk_objective import cpp_stalk_objective_model

model = cpp_stalk_objective_model(
    source, state, probes, views, channel_coefficients=(0, 0, 1, 0))
output = infer_model(model, source, stalk_transport_query(state, query_field))
```

Channel coefficients are ordered T, raw G, F and C. They and `mass` must be
nonnegative integers or `Fraction` values. Outputs retain both evidence and
conditional field families in `execution_receipt`. An objective choice does not
identify a map from observations. Additional training appends observed fields and
recomputes the conditional family. Model records, stores, RCQL, prediction workers
and System previews preserve this distinction. The reduced parameter action can
fill during exact solving and remains subject to the model's action admissions.
`stalk_objective_field(output, interpretation="evidence")` requires a whole field
fixed by the observations. Use `interpretation="conditional"` to explicitly
select the objective family. Unresolved field directions reject; ragged stalks
require an original `row`.

### Parameter relation certificates

`stalk_parameter_relations_query` checks declared equations on the complete
observed map family. Each relation is `(name, terms, expected)`, with terms
`(entry_index, coefficient)` on the original `parameter_slots`.

```python
from rexgraph.cpp_stalk_parameter_relations import (
    stalk_parameter_relations_query, stalk_parameter_relation,
)

query = stalk_parameter_relations_query(
    state, (("shared-gain", ((0, 1), (1, -1)), 0),),
    selected=("shared-gain",))
output = infer_model(state, source, query)
relation = stalk_parameter_relation(output, "shared-gain")
```

`forced` means every observed solution satisfies the equation. `excluded` means
none does. `open` means both satisfying and nonsatisfying solutions remain.
The output retains the residual origin and every original free direction,
including zero directions. `conditional_family` in its receipt reports whether
the selected equations are jointly consistent and retains their complete
intersection separately from the original evidence family. The query does not
change the model or identify its source carriers. For version 2, certificates
are conditional on the declared parameter basis. Model records, stores, RCQL
`MODEL_INFER`, prediction workers and System previews support these queries.

`stalk_context_query(state, target_rows, frame_atlas)` acquires complete typed
star comparisons from the retained training observations. Each atlas entry is
`None` or `(common_frame_name, axis_permutation)` and explicitly maps one original
stalk to that common frame. The atlas covers grades 0 and 1. Equal axis names or
dimensions do not supply a map. `stalk_context_hypotheses(output)` returns the
original cell and occurrence maps, supporting observations, complete entry
equations and whole block statuses. Select a conditional family with
`selected_blocks=((origin_row, target_row),)`. An open block remains a hypothesis,
including when its target has no observed field. This profile preserves ordered
occurrences and compares corresponding fixed and variable support; it does not
infer new relation types or roles from text.

`stalk_context_views_query(state, target_rows, view_atlas)` compares stars whose
original stalk spaces differ. Each entry is `None` or
`(common_carrier_name, coordinate_keys, accession_entries, reconstruction_entries)`.
Map entries are `(row, column, integer_or_Fraction)`. Reconstruction composed with
accession must be the identity on every original stalk. The native query computes
the complete intersection of the declared input images and compares realized
boundary responses there. Acquired blocks retain that input frame, both original
input reconstructions and full response shapes. Unshared inputs remain unresolved
in the original evidence family. Select blocks and read their certificates with
the same `selected_blocks` and `stalk_context_hypotheses` interfaces.

`stalk_context_diagrams_query(state, target_patches, view_atlas)` compares connected
groups of at least two original relation rows. For example, use
`target_patches=((row_a, row_b),)` when those rows share an original boundary.
The native query retains compatible complete star maps and rejects conflicting
shared identities. `stalk_context_hypotheses(output)["diagrams"]` includes each
whole map, its shared boundaries and its joint evidence status. Select a complete
conditional map with `selected_diagrams=(((origin_a, row_a), (origin_b, row_b)),)`.
Open groups remain prospective. Their equations and all original unresolved
directions roundtrip through the existing model records and stores.

`stalk_context_rules_query(state, training_rows, target_rows, frame_atlas)` learns
sharing rules from independent original map entries. By default, each rule needs
two other observed contexts whose complete transport blocks agree. Contradictory
or unresolved compatible training blocks prevent admission. Target rows must be
outside the training family. Transfer to an unobserved target is conditional on
the observed context rule; the original model and its free directions remain.

```python
from rexgraph.cpp_stalk_parameter_relations import (
    stalk_context_rules_query, stalk_context_hypotheses, stalk_context_rule_response,
)

query = stalk_context_rules_query(state, training_rows, target_rows, frame_atlas)
output = infer_model(state, source, query)
rules = stalk_context_hypotheses(output)
response = stalk_context_rule_response(output, query_field)
```

The response retains complete boundary and occurrence families, including any
unresolved directions. Rule outputs store nonzero residual coefficients with
their original relation and component coordinates. The receipt retains the full
logical shape and exact proof. No supplied parameter basis is accepted.

For different original stalk dimensions, use
`stalk_context_view_rules_query(state, training_rows, target_rows, view_atlas)`.
Declare faithful rational accession and reconstruction maps in `view_atlas` as
for `stalk_context_views_query`. Each target comparison needs enough agreeing
independent peers that cover its complete shared input image. Directions outside
the overlap remain unresolved. Read and apply the result with
`stalk_context_hypotheses` and `stalk_context_rule_response` as above. Acquisition,
joint evidence checks, input coverage and field assembly run natively. The
parameter relation owner reuses the acquired native proof.

`compile_stalk_context_pattern` exports one admitted training star as a compact
native inference model. The model retains exact occurrence weights, their
unresolved directions and SHA256 references to its source and training state.
Full provenance requires those external archives. Inputs outside the shared
domain of the agreeing peers reject explicitly.

```python
from rexgraph.cpp_stalk_pattern import compile_stalk_context_pattern
from rexgraph_cpp import PreparedStalkTransportPattern

pattern = compile_stalk_context_pattern(state, source, query, training_rows[0],
    model_namespace=904, coordinate_ids=(100, 101, 102))
weights = pattern.inspect()
image = pattern.archive()
restored = PreparedStalkTransportPattern(image, model_namespace=904)
response = restored.response(common_input)
```

The three coordinate IDs declare the domain basis, ordered weights and free
parameter basis. They must be distinct and separate from the common atlas
frames. The native archive owns the pattern complex and its tensors. It does
not retain the original observations or source bytes. `inspect` marks each
identified weight beside the complete affine parameter directions.

`compile_stalk_context_pattern_bank` stores selected training stars together.
Its joint weight family preserves correlations between unresolved parameters.
The archive retains shared boundary identities once.

```python
from rexgraph.cpp_stalk_pattern import compile_stalk_context_pattern_bank
from rexgraph_cpp import PreparedStalkTransportPatternBank

bank = compile_stalk_context_pattern_bank(state, source, query, training_rows[:2],
    model_namespace=905,
    coordinate_ids={"weights": 101, "parameters": 102, "domains": (100, 103)})
restored = PreparedStalkTransportPatternBank(bank.archive(), model_namespace=905)
response = restored.response({0: first_input, 1: second_input})
gluing = restored.glue({0: first_input, 1: second_input})
```

Input keys select positions in the stored bank. `inspect` returns the joint
weight family, input domains, original cell identities and provenance references.
Column training observations are read stalkwise during context comparison and
pattern compilation. A bank retains only its selected original stars and their
joint observable weights. It can serve local sparse inputs without loading the
training source or observations.

`bank.select(patterns, model_namespace=..., coordinate_ids={"weights": ..., "parameters": ...})`
creates a smaller bank from stored patterns. Supply ordered pattern indices,
a new namespace and new weight and parameter coordinate identities. The result
retains complete original stars and their joint weight family. Source and
evidence digests and fit counts remain unchanged.

`compile_stalk_context_pattern_scope(state, source, query, source_namespace=...)`
retains all observed original stars and their common input images. Load its bytes
with `PreparedStalkTransportPatternScope` and call `scope.coverage(bank)` to read
the stored patterns covering each original star. Empty entries identify omitted
stars or uncovered input directions. Coverage checks complete typed boundaries
and exact input subspaces. It does not establish output agreement or unseen
branch completeness. Source and evidence digests reference external archives.
`scope.for_bank(bank, source_namespace=...)` exports all observed branches sharing
the stored patterns' primary types and common input carriers into a new scope.

For a serving model, use the same selection through the model interface:

```python
from rexgraph.cpp_stalk_pattern_model import select_stalk_pattern_bank_model

local_model, local_source = select_stalk_pattern_bank_model(
    model, source, (0, 2), model_namespace=907,
    coordinate_ids={"weights": 1001, "parameters": 1002},
)
```

Input keys use the selected bank's positions. Selection reads the parent bank
once; subsequent inference and cold loading require only the selected archive.
Keep participating stars together when a task depends on their joint parameters.

`glue` checks shared boundary readings and optional declared restriction maps.
A compatible result contains the shared section once, the separately assembled
boundary response and the remaining exact parameter family. An inconsistent
result contains the complete defect family and a contradiction witness.
Comparison `left` and `right` select retained occurrence positions. Their
`left_frame`, `right_frame` and `interface_frame` IDs declare the carriers;
`left_map` and `right_map` supply exact matrix shapes and sparse entries.
`PreparedRelationGuards` can read the bank archive. The `pattern_source` returned
by `inspect` contains RFCX source bytes.

Use a bank archive as a source for the shared serving model API:

```python
from rexgraph_cpp import NativeSource
from rexgraph.cpp_stalk_pattern_model import (
    cpp_stalk_pattern_bank_model, stalk_pattern_bank_query,
    stalk_pattern_bank_result, inspect_stalk_pattern_bank_model,
)
from rexgraph.model_runtime import infer_model
from rexgraph.io.model_record import model_record, save_model

source = NativeSource(bank.archive())
model = cpp_stalk_pattern_bank_model(source)
query = stalk_pattern_bank_query(model, {0: common_input}, operation="glue")
output = infer_model(model, source, query)
family = stalk_pattern_bank_result(output)
weights = inspect_stalk_pattern_bank_model(model, source)
save_model("patterns.rcbd", model_record(source, model))
```

Saved models retain the bank archive and serving declaration. Training batches
remain external. `MODEL_INFER` accepts the declared query; `MODEL_VALUES` returns
the complete response or gluing family. Unknown directions and incompatible
sections remain explicit. A bank serves its learned weights; further learning
uses the original transport model before exporting a new bank.
Each field's `directions` spans that field's own response family. Gluing's
`parameter_origin` and `parameter_directions` describe the conditioning of the
bank's joint parameter frame. Separate field frames do not identify their free
coordinates with one another.
For System query requests, `system.serialize.json_value(query)` renders exact
rationals and integers for JSON parameters. The serving provider checks the
model and source identities when decoding this declared query format.

`replay_stalk_transport_model(state, source)` explicitly replays original batches
after a native implementation change and checks the complete retained family.
`replay_graded_stalk_transport_model` provides the same check for graded models.

### Graded incidence transport learning

`cpp_graded_stalk_transport_model` learns transport entries at one selected
original `grade`. Supply named `stalk_axes`, complete `fixed_maps` at every grade,
`parameter_slots` and lower `training_rows`. The fixed tower must satisfy the
complete chain family. The learned family satisfies both adjacent constraints;
transport families at other grades remain fixed.

```python
from rexgraph.cpp_graded_stalk_transport import (
    cpp_graded_stalk_transport_model, graded_stalk_transport_observation_batch,
    graded_stalk_transport_query, graded_stalk_transport_maps,
)
from rexgraph.model_runtime import train_model, infer_model

state = cpp_graded_stalk_transport_model(
    source, stalk_axes, fixed_maps, parameter_slots, training_rows, grade=2,
)
batch = graded_stalk_transport_observation_batch(state, input_field, annotations)
state = train_model(state, source, batch)
output = infer_model(state, source, graded_stalk_transport_query(state, query_field))
```

Optional `parameter_space` and `parameter_basis` declare exact named templates.
Fit receipts report chain, observation and combined ranks and retain all free
directions. `graded_stalk_transport_maps` requires explicit `free_parameters`
when the maps remain unresolved. A direct native owner also provides
`PreparedStalkBoundary.fit_transports(grade, parameter_slots, observations)`.
Its `append` method shares the fixed chain and fits only new whole fields.
`execution_stats` reports reused observations and actual guarded work.

For fixed maps that require completion, use `cpp_affine_stalk_transport_model`
with the same arguments. This declares adapter version `2` and delegates the
complete affine chain constraints to native execution. Inconsistent fixed maps
reject. The original constructor retains its strict baseline requirement.

The direct `PreparedStalkDeclaration(source.archive, stalk_axes, fixed_maps)`
owner provides `fit_transports` without exposing boundary execution. Its fitted
family retains the complete `target`; the first `chain_shape[0]` entries are the
possibly nonzero chain right hand side. Materialize the fitted maps with explicit
values for every unresolved parameter, then pass those maps to
`PreparedStalkBoundary` for strict boundary execution. Model records preserve the
adapter version, original declaration, complete target, evidence and native pin.

RCQL uses `MODEL_INIT("cpp_graded_stalk_transport", config)` and the existing
training, inference and model information operators. Set
`config["affine_completion"] = True` to select adapter version `2`; the default
selects the strict baseline constructor. Model records retain the
original source and batches and support continued fitting after restore.

`PreparedLexicalRoutes(guard_owner, spellings)` prepares exact lemma and form
routes from original signed dictionary columns. Each spelling supplies `row`,
`raw`, `uses`, `encoding_rows` and `spans`. Verify literal byte fields and token
maps before supplying these accessions. `lookup(raw, spans)` retains every
matching sense path, including overlapping phrases and repeated occurrences.
`bind_patterns(pattern_guard_owner)` checks complete original pattern stars
before associating their learned fields with dictionary routes.

`query_guard.encoding_accession(dictionary_guard, query_rows, dictionary_rows)`
checks complete closed `use-encoding` carriers across the two named owners.
The result includes every original lower, relation and occurrence pair and both
source receipts. It preserves ordered unit encoding flags and checks the whole
linear chain map. Each owner's full type signature remains separate. Original
byte charts must be verified before supplying the row pairs.

Pass `observed_inputs=True` to `stalk_context_view_rules_query` to compare exact
spans of retained input fields. The original stalk carriers and unidentified
transport parameters remain complete. The default compares the full declared
input carrier. Both modes require independent agreeing peers before transfer.

`stalk_context_rule_restriction(output, observations)` checks complete proposed
fields jointly against a learned conditional family. Each observation supplies
`input`, `target` and a Boolean `observed` mask on original lower stalks. Results
report `forced`, `open`, `excluded` or `inconsistent-base`, retain the conditional
parameter family and include an exact contradiction witness when incompatible.
`validate_stalk_context_restriction` replays a saved result against its output.
The direct native owner exposes `restrict_fields(observations, selected=())`.

For repeated calls, construct `PreparedStalkContextRules(output)` once. Its
`apply(field)` and `restrict_fields(observations)` methods reuse the owning
native proof after complete preparation. `receipt` retains the original model,
source and selected rule identities. Model records still store the existing
model state, output and ordinary metadata.

`compile_stalk_context_pattern` and `compile_stalk_context_pattern_bank` accept
observed input queries. Their serving archives retain named common carriers,
observable weights and joint parameter directions. Inputs outside the stored
common span raise `ValueError`. Original source and evidence digests refer to
the separate training audit. Direct native learning accepts `observed_inputs=True`.

Use `response_sparse`, `response_sparse_batch` or `glue_sparse` on a prepared
bank to provide complete fields without constructing dense input vectors:

```python
from fractions import Fraction

field = dict(profile="complete-sparse-stalk-field-v1", frame=2, size=65536,
             entries=((0, Fraction(2, 3)), (65535, Fraction(-3, 7))))
family = bank.response_sparse({0: field})
families = bank.response_sparse_batch(({0: field}, {0: field}))
```

The frame and size must match the pattern's declared common input carrier.
Entries must have increasing distinct indices and nonzero exact coefficients.
Omitted coefficients are known zero. Fields outside the evidenced input image
reject. A batch shares one native work budget and returns only after all fields
pass. Use `stalk_pattern_bank_sparse_query` for the same inputs through the
shared model API, saved models, RCQL and System queries.

Use `response_sparse_candidates({pattern: field})` to evaluate alternatives
separately under one native work budget. Each result names its original pattern
and contains either the complete response family or `supported=False` with
`response=None` for an input outside that pattern's evidenced domain. Invalid
frames, malformed fields and resource failures reject. The shared model API
accepts the same declaration through
`stalk_pattern_bank_sparse_query(state, inputs, operation="candidates")`.

Use `response_sparse_program(nodes)` to pass fields between learned stages in
one native action. Each stage declares `name`, `pattern`, `input` and `edges`.
An external input uses the complete sparse field format above. A dependent stage
has `input=None` and incoming maps with `node`, `occurrence`, `source_frame`,
`target_frame` and `map`. Each map declares its exact `shape` and sparse `entries`.
The original occurrence must belong to the preceding stage's pattern. Its
unscaled field supplies the map input. Several incoming maps explicitly sum
their full fields on the destination carrier.

Results retain every stage's input, complete response and shared parameter
directions. Status is `supported`, `outside-domain`, `open-input` or `blocked`.
An unresolved mapped input is retained without assigning its unknown parameters.
Original correlations can cancel before a later stage executes. These maps
declare field composition; they do not certify a chain map or compatible gluing.
Use `stalk_pattern_bank_program_query(state, nodes)` for the same operation
through Core, RCQL, Agent, System and model records.

`inspect()["provenance_bytes"]` reports the exact encoded provenance record size.
Digest bytes and population counts can change this size while learned tensors
remain unchanged.

## Whole field programs

`PreparedFieldProgram` owns a fixed exact program on one source publication.
Its closed declaration uses profile `rexgraph.relational-field-program.v1`,
`key`, `revision`, `differential`, `stalks`, `occurrences` and ordered `nodes`.
Every original cell declares its ordered basis directions and positive metric.
Every original occurrence declares its complete rectangular map and either
`coefficient_independent` or `realized_incidence` role. Matrix records contain
`shape` and sparse `(row, column, integer_or_Fraction)` entries.

Nodes declare `key`, `kind` and `grade`. An `input` node receives a complete
native tensor. A `boundary` node names its preceding `input`; a `declared_map`
node also declares its exact `map`. Native preparation checks dimensions,
basis identity and the applicable complete adjacent boundary laws.

```python
from rexgraph.cpp_relational_field_program import (
    cpp_relational_field_program_model, field_program_query,
    read_field_program_result,
)
from rexgraph.model_runtime import infer_model
from rexgraph.io.model_record import model_record, save_model, load_model, read_model_record

# source is an owning NativeSource; declaration specifies the complete program.
# field_bytes encodes a complete supported tensor on input node 1's stalk carrier.
state = cpp_relational_field_program_model(source, declaration)
query = field_program_query(state, ({"node": 1, "use": 17, "field": field_bytes},))
output = infer_model(state, source, query)
result = read_field_program_result(output, state=state)
metadata = result.fields(max_records=64)
native_field_bytes = result.field_archive(2)
selected = result.select(2, ((0, 0),), coordinate_limit=32)

record = model_record(source, state)
record.attach_metadata(1, 0, "response", output)
save_model("program.rcbd", record)
restored, restored_source = read_model_record(load_model("program.rcbd"))
```

`output.field_archive` retains the complete native result, program and input use
identities. Result restoration recomputes every intermediary and occurrence
field. `result.select` returns complete selected stalks with their original
basis, directions and metric, subject to its coordinate bound. Whole field
outputs require explicit node/stalk selection before producing Python values.

RCQL supports `MODEL_INIT`, `MODEL_INFER`, `MODEL_INFO` and `MODEL_RECORD` for
adapter `cpp_relational_field_program`. `MODEL_VALUES(output)` inspects bounded
field metadata; `MODEL_STALK(output, [grade, cell], node=key,
max_coordinates=256)` selects a complete stalk. Both validate the selected
source. System summaries retain archive receipts without expanding field
coefficients. Agent workers can call `infer_model` through `Hive.add_worker`.

This version supports fixed create, prepare and infer operations. Training and
parameter transport reject. Native execution requires the recorded implementation
receipt; retain the corresponding runtime for pinned models. Portable model
output schema 3 stores whole field archives, while legacy schema 1/2 bytes remain
unchanged. Bounded result inspection does not provide independent local inference.

`cpp_accession_field_program_model` selects adapter/provider version 2. Its query
retains actual original byte/span/source-use assets and explicit faithful input
accessions. `NativeInputUseAsset` requires its complete native source archive and
the profile `rexgraph.relational-input-use-asset.v1` with `key`, `channel`,
`encoding`, original `bytes` and ordered `uses`. Each use supplies `key`, original
`cells` and `occurrences` as `(grade, stable_identity)` pairs, and ordered byte
`spans`. Recovery keeps empty, overlapping, repeated and discontinuous intervals.
An encoding label preserves supplied context; numeric meaning remains separately
declared field evidence.

```python
from rexgraph_cpp import NativeInputUseAsset, NativeFieldSection
from rexgraph.cpp_relational_field_program import (
    cpp_accession_field_program_model, accession_field_program_query,
)

asset = NativeInputUseAsset(original_source.archive, asset_declaration)
state = cpp_accession_field_program_model(source, declaration)
query = accession_field_program_query(state, ({
    "asset": asset.archive(), "accession": accession_declaration,
    "use": 17, "field": original_field_bytes,
},))
output = infer_model(state, source, query)
result = read_field_program_result(output, state=state)
interface = result.interface(interface_declaration)
reading = interface.observe(view_declaration)
section = NativeFieldSection((reading,), kind="compatible")
restored_section = NativeFieldSection.from_archive(section.archive())
```

An input accession declares profile `rexgraph.relational-input-accession.v1`,
`key`, target input `node`, original source `grade`, complete original `stalks`,
full `instances` maps (`cells` and `occurrences` per grade), `accession`,
`reconstruction` and explicit boolean `isometric`. Native admission checks RA=I,
complete support, actual instance maps, both metrics and the admitted image.
The result retains original input assets, fields and source publication references.

An interface declares profile `rexgraph.relational-field-interface.v1`, `key`,
`branch` and ordered `(node, grade, cell_index)` `cells` forming a closed under the boundary
patch. A view declares profile `rexgraph.relational-field-view.v1`, `key`, `basis`,
ordered `directions`, A/R as `accession`/`reconstruction`, and separately declared
`physical` matrix or `None`. `interface.observe(view)` derives the complete view;
`interface.admit(view, native_field_bytes, status=...)` accepts explicit `derived`,
`observed` or `prospective` readings on its admitted image.

Compatible sections require the actual same interface owner and agreeing original
fields. Equal bytes, keys or numerical encodings do not supply that ownership.
`kind="contribution_sum"` performs a separately declared sum and retains every
reading. Section archives restore one shared interface with complete parent/input
assets and replay all original fields and views. `section.select(cells,
coordinate_limit=...)` reads bounded complete original stalks. RCQL selects version 2
with `MODEL_INIT("cpp_relational_field_program", {"source_backed": true, ...})`.
This profile remains fixed source inference; affine training, generalized dependency
availability, source extension and independent local serving require their own APIs.

`cpp_field_learning_model` selects adapter/provider version 3. Declare a differential
program with one Boundary node at the learned grade and explicit coefficient
independent maps for every original flag. The parameter profile
`rexgraph.relational-field-parameters.v1` supplies positive `key` and `revision`,
the Boundary `node`, ordered original `(flag_index, row, column)` `variables`,
the exact coordinate `basis` matrix and distinct named `parameters`. The basis
declares actual shared changes to those original maps. Source, stalk bases,
metrics, node graph and output axes remain fixed during training.

```python
from rexgraph.cpp_relational_field_learning import (
    cpp_field_learning_model, field_program_batch, field_learning_query,
    read_field_learning_result, evaluate_field_program,
)
from rexgraph.model_runtime import train_model

state = cpp_field_learning_model(source, declaration, parameters=parameters)
batch = field_program_batch(state, ({
    "key": 1, "inputs": original_input_rows,
    "target_asset": measurement_asset.archive(), "target_use": 23,
    "target": complete_target_field_bytes, "observed": ((0, 0), (0, 1)),
},))
child = train_model(state, source, batch)
query = field_learning_query(child, independent_input_rows)
output = infer_model(child, source, query)
family = read_field_learning_result(output, state=child)
whole = family.field_archive(2)
maps = family.evidence().occurrence_map(1, 0).inspect(max_records=32)
residual = family.evidence().residual(0).field_archive()
prospective = evaluate_field_program(output, complete_free_values, state=child)
```

Each observation retains complete native input fields and target measurements,
their actual original assets and source cutoffs, and an ordered whole stalk mask.
The target field is explicitly declared on the learned node's model carrier;
target bytes do not imply a numeric encoding. Unobserved measurement directions
remain in the archive. One append produces one immutable child with native exact
parent relations and the original coordinate order.

Inference retains one common named free parameter carrier through all complete
node and occurrence fields. `field_archive` returns native `particular` tensor
bytes and `directions` matrix bytes; the particular is a representative. Selected
stalks report the particular, complete directions and exact identified flags.
Inactive and zero participation maps retain unresolved alternatives. Prospective
evaluation requires every free value explicitly and preserves the complete family,
evidence, original inputs and chosen values. Cold model and result replay checks
whole native fields, evidence, dependencies and parent seals.

RCQL selects version 3 with explicit `source_backed: true` and `learning: true`,
supports `MODEL_BATCH`/`MODEL_TRAIN`, and uses the existing whole field inspection
routes. System summaries expose evidence counts and receipts. Agent capability
workers call the common `infer_model` route. General dependency availability,
conditional metric objectives, multiple learned boundary uses and source/basis
extensions require separately declared profiles.

`cpp_field_comparison_model` selects the fixed adapter/provider version 4. Its
comparison declaration supplies profile `rexgraph.relational-comparison-forms.v1`,
positive `key`/`revision`/`grade`, explicit `participation` (`share` or
`legacy_count`), and complete ordered `weights` with original `(grade, cell_index)`
`cell` and exact rational `value`. Every original flag at that grade requires an
explicit coefficient independent Phi map, including inactive and zero flags.
Choose the actual input `node` at that grade and channel `T`, `rawG`, `C` or `F`.

```python
from rexgraph.cpp_relational_field_comparison import (
    cpp_field_comparison_model, field_comparison_query, read_field_comparison_result,
)

state = cpp_field_comparison_model(
    source, declaration, comparison=comparison, node=1, channel="C",
)
output = infer_model(state, source, field_comparison_query(state, original_input_rows))
comparison_result = read_field_comparison_result(output, state=state)
whole_output = comparison_result.field_archive("output")
original_program_result = comparison_result.input()
original_flag_field = comparison_result.occurrence_field_archive(0)
pair = comparison_result.pair(0, 2, channel="C")
pair_observation = pair.inspect()
stalk = comparison_result.select(((1, 0),), coordinate_limit=32)
```

The source backed parent retains all original program intermediates, inputs,
assets and accessions. Comparison results retain complete input/output, signed
and unsigned shared boundary fields, every original Phi field, bases and metrics.
Cold replay rebuilds the comparison against the actual decoded parent and checks
the complete native action. Pair readers retain the original requested flag order
and actual parent without serializing it implicitly; weight/admission state and
quadrance remain explicit diagnostics. A zero contribution preserves the residual.
Ordinary actions use native factored sums. `PreparedFieldComparison.operator_matrix`
requires an explicit export bound.

RCQL selects version 4 with `source_backed: true` and `comparison_fields: true`.
`MODEL_VALUES` inspects comparison metadata, and `MODEL_STALK` selects the declared
comparison output grade without a node argument. Training rejects. Unresolved
affine families require a separately declared conditional comparison or an
explicit prospective materialization retaining their evidence and chosen values.

## Available joint field families

`PreparedAvailableLiteralFieldFamily` retains the actual available evidence
members, their ordered training readers and one complete common parameter
family. Supply independently published origins and an explicit
`NativeAvailabilityContext` when constructing, restoring or using it. Results
retain the same whole node and occurrence fields through inspection and a
prospective evaluation.

```python
from rexgraph_cpp import PreparedAvailableLiteralFieldFamily

model = PreparedAvailableLiteralFieldFamily.from_archive(
    model_bytes, context, limits=exact_action_limits, max_bytes=archive_limit,
)
result = model.execute(member_index, independent_reader_rows, context)
whole = result.field_archive(node, context, occurrences=True)
packet = result.prepare(context)
with open("result.packet", "wb") as stream:
    packet.stream(stream.write, context)
```

`NativeAvailableLiteralJointFieldResult` and
`NativeAvailableLiteralJointFieldEvaluation` also restore with independent
context. Each operation rechecks the retained dependency owners, including
prepared packet reads and streaming. Streams deliver at most 64 KiB per callback
and reject short writes. A packet retains its native owners after the Python
model wrapper is released.

The exact action policy must match the archived model. Reader and archive bounds
are explicit resource limits; increasing them does not change that policy.
Inspection bounds reject oversized results rather than truncate them. Prospective
settings and reader specific completions require separate published origins.

`replay_joint_reader_input` rebuilds a literal or known tensor graph input on an
actual supplied program only when the complete program and reader images match.
`replay_available_joint_family` replays original evidence through available
program, root and append operations, preserving append batches and actual parents.
Neither operation creates publication origins or grants. Source, basis, map and
metric changes require their own declared transport.

## Available source backed fields

`NativeValueOrigin` is an explicit value publisher: supply its key, original native
history, authentic publication cutoff, dependency role and complete native payload.
`NativeAvailabilityContext` takes actual origin owners with each admitted namespace
and sequence limit. Input value publication is separate from byte asset publication.
Use `field_input_origin_payloads` to obtain complete passive native input payloads
before constructing an accession. Publishing an origin is a separate operation
from inference and restoration.

`PreparedAvailableFieldProgram` checks explicit Program/Maps/Metrics origins and
uses `execute_inputs` with complete source backed declarations and four origin keys
per input. Every call takes an independently supplied context. Its archive uses
RFCVP v1; `NativeAvailableFieldResult` uses RFCVR v1. Their `from_archive` methods
require the context and validate actual content and publication receipts before
native actions. `admit(context)` rechecks a retained owner under current limits.

```python
from rexgraph.cpp_relational_field_admission import (
    cpp_available_field_program_model, available_field_query,
    read_available_field_result,
)
from rexgraph.model_runtime import infer_model

state = cpp_available_field_program_model(
    source, available_program=available_program, execution_context=context,
)
query = available_field_query(state, original_input_rows_with_origin_keys)
output = infer_model(state, source, query, execution_context=context)
result = read_available_field_result(output, state=state, execution_context=context)
whole_fields = result.result()
```

The model stores complete origin obligations; the saved model does not supply its
own execution authority. Pass `execution_context` to model save/load and record
restoration. RCQL selects version 5 with `source_backed: true` and
`available_fields: true`; provide its actual `available_program` and
`execution_context` in creation configuration, and pass `execution_context` to
`MODEL_STATE`, `MODEL_INFER`, `MODEL_VALUES`, `MODEL_STALK` and `MODEL_INFO`.
Agent capability callbacks call `infer_model` with their configured context.
This fixed program profile does not train. Source extension, evidence admission
and independently served local solves require their separate declared profiles.

## Available whole field learning

`field_evidence_origin_payload` binds the original program, parameter declaration
and exact policy for an explicit Evidence publication. Construct
`PreparedAvailableFieldEvidence(available_program, parameters, origin_key, context)`
after that origin is admitted. Use `field_target_origin_payload` for complete
TargetValue publication; TargetAsset has its own origin. Each observation retains
original input rows, target asset/use/field, whole stalk mask and target origin keys.
Native `append_inputs` checks the entire suffix before input actions or fitting.

```python
from rexgraph.cpp_relational_field_evidence_admission import (
    cpp_available_field_learning_model, available_field_program_batch,
    available_field_learning_query, read_available_field_learning_result,
)
from rexgraph.model_runtime import train_model, infer_model

parent = cpp_available_field_learning_model(
    source, available_evidence=root_evidence, execution_context=context,
)
batch = available_field_program_batch(
    parent, source, original_observations_with_origin_keys, execution_context=context,
)
child = train_model(parent, source, batch, execution_context=context)
query = available_field_learning_query(child, independent_input_rows_with_origin_keys)
output = infer_model(child, source, query, execution_context=context)
family = read_available_field_learning_result(output, state=child, execution_context=context)
whole_fields = family.result()
```

The complete correlated parameter family remains available through
`family.evidence().evidence().inspect()`. A global prospective setting uses the
native evidence owner's `parameter_setting_payload(values)` and a separately
published Prospective origin. `evaluate_available_field_program` takes that
origin key and context; it preserves the family and each independently admitted
held input use. A selected setting is prospective data.

Available evidence, families and selected evaluations use RFCVE/RFCVF/RFCVQ v1.
Cold readers require independent context and replay complete original facts.
RCQL selects provider version 6 with `source_backed`, `available_fields` and
`learning` all true and an actual `available_evidence` root. Pass context to
training, inference, inspection and restoration. Ordinary training preserves
the original source, coordinates and axes; structural extension requires an
explicit separate source/model operation.

### Solving a local response

A local response still depends on its global interface. Supply that interface
and its reconstruction when requesting a smaller patch, so the patch can be
interpreted as a view of the complete response.

Exact local responses accept an admitted fixed field output or an explicitly
selected available affine evaluation. Publish the payloads returned by
`local_field_origin_payloads(rhs, node, declaration, readout)` with Maps,
Interface and Reconstruction roles, then pass their keys and independent context.
A declared harmonic frame also needs a Maps origin; a derived frame uses gauge
key zero. The declaration names grade, Green or harmonic inverse, exact lambda,
coordinate identities and physical interior indices. The readout retains an
ordered patch closed under the boundary in every grade, together with faithful
accession/reconstruction maps and named directions.

```python
from rexgraph.cpp_relational_field_local import (
    submit_available_field_solve, read_local_field_result,
)

run = submit_available_field_solve(
    output, node, declaration, readout, origin_keys, execution_context=context,
)
local_output = run.wait(execution_context=context)
local = read_local_field_result(local_output, execution_context=context)
whole_response = local.field_archive("global")
interface_response = local.field_archive("interface_response")
patch = local.field_archive("patch")
represented = local.field_archive("visible")
```

`run.cancel()` requests native stop and joins before returning. Work keys,
generations and logical worker/buffer/result demand are explicit through `work`.
Bounded atomic native phases do not promise a stop latency or physical memory
bound. Result reads require current context after completion. `local.info()`
retains complete boundary/metric/gauge/augmented/Schur and readout matrices;
`local.rhs()` retains original uses, complete fields and any learned family and
prospective setting. Original stalk selection is bounded explicitly.

RFCVL v1 packets disclose complete global context and replay with the separate
model file absent. They retain the whole global response, interface, patch and
reconstruction. Use the pinned matching runtime and independent context for
cold reads. Portable Core output packing retains packet bytes without the
optional native provider; execution requires the provider. RCQL `MODEL_LOCAL`
takes output, node, declaration, readout, origins and execution context.
`MODEL_VALUES` and `MODEL_STALK` inspect its complete metadata and original
stalks. Capability workers can call the same Core functions.

### Extending the source

Training on more evidence keeps the existing source and coordinates. Adding a
relation, stalk direction or grade changes the model's declared space and needs
a structural child with a map back to its parent.

Faithful source extensions use a separate version 7 model. Construct an owning
`PreparedAvailableFieldExtension` from an admitted fixed program or an explicitly
selected available learned evaluation, an independently admitted target program,
and a declared source map with inclusion and reconstruction for each stalk. Publish
`field_extension_origin_payload` with the declared Evidence or Prospective role.
The inclusion must preserve every original boundary, flag map and program action;
metric pullback is checked when requested. Reconstruction is not an inferred
Green or Hodge transport.

```python
from rexgraph.cpp_relational_field_extension import (
    cpp_available_field_extension_model, field_extension_query,
    read_field_extension_result, evaluate_field_extension,
)
from rexgraph.model_runtime import infer_model

child = cpp_available_field_extension_model(
    target_source, parent=parent_model, available_extension=extension,
    execution_context=context,
)
query = field_extension_query(child, original_input_rows)
output = infer_model(child, target_source, query, execution_context=context)
family = read_field_extension_result(output, execution_context=context)
whole_field = family.field_archive(node)
flags = family.field_archive(node, occurrences=True)
relationship = family.model().info()
```

The child retains the complete parent checkpoint and native extension lineage,
original input source and independently admitted target source. New relations,
basis directions and higher grades belong to this explicit structural child.
Ordinary training cannot extend the source. Whole particular and direction fields,
supported zeros, original ordered occurrences and the common completion carrier
remain inspectable before any completion.

Publish `family.completion_payload(values)` as an input specific Prospective origin,
then pass every declared completion value and that origin key to
`evaluate_field_extension(output, values, origin, execution_context=context,
state=child)`. Completion never supplies omitted values or reuses another input's
choice. Its output is also an owning RHS for the local solve functions above.
RCQL `MODEL_EXTEND` requires an explicit target Binding; `MODEL_COMPLETE` selects
the prospective input completion. Both require separate current context. Complete
RFCWX/RFCWY/RFCWZ v1 archives retain the native relationships and replay with
independently supplied authority. Incompatible wrapper implementations require
their pinned runtime.

## Finite sources and interpretations

A word occurrence can admit several interpretation branches even when another
occurrence has the same bytes. A finite source retains those candidate symbols
and their use bindings. A field map associates supported assignments with
numerical targets; unsupported assignments remain visible as alternatives.

Raw finite carriers use `NativeFiniteSource(asset, declaration)` with the actual
input use asset, opaque symbol definition bytes, ordered variable domains, closed
charts and original use bindings. Symbols stay separate from numerical coordinates.
`source.sections()` returns the complete compatible native family and candidate
count; `select(begin, count)` performs bounded inspection. Empty compatible
families and refused resource admissions are distinct outcomes.

`PreparedFiniteFieldMap(source, targets, declaration)` retains ordered actual
accession programs or complete `PreparedFieldEvidence` targets. Each rule declares
its binding/symbol dependencies and actual admitted `NativeAccessionInput` owners.
An unsupported rule retains its reason and supplies no numerical target or field.
The native executor retains every compatible alternative, including unmapped
assignments and distinct branches with equal numerical fields.

```python
family = finite_map.execute()
branches = family.select(0, count)
branch = branches[index]
meaning = branch.inspect()
whole_fields = branch.result()  # fixed, affine, or None for unresolved alternatives
selected = family.evaluate(ordinal, complete_free_values)
actual_response = selected.result()
original_alternatives = selected.family()
```

Fixed branches require an explicitly empty free vector. Affine branches require
every actual common free axis, including an empty vector when already identified.
Selected values retain the whole original finite family, ordinal, assignment,
rule, uses, spans and full fixed or affine response. Original map families and
whole fields remain inspectable through the branch's actual evidence owner.
RFCQS/RFCQM/RFCQB/RFCQC archives replay complete native witnesses and reject
foreign live owners; cold adoption is explicit.

These raw APIs confer no availability or publisher authority. They do not train
interpretation tables, supply missing values, decode language, or promote finite
choices into existing admitted local RHS types. Opaque archive bytes may be packed
and copied through Core; admitted service integration requires its available
finite provider and separate current authority.

### Available finite maps

Use the available wrapper when inspection and execution must respect publication
cutoffs. Archive contents describe dependencies; an independently supplied
context determines which dependencies can be used at the current cutoff.

Available finite maps require an independently supplied `NativeAvailabilityContext`.
`PreparedAvailableFiniteSource` binds the original asset and a classified constraint
receipt. `PreparedAvailableFiniteFieldMap` binds every target and raw rule input,
plus separate Maps and relationship receipts. Input submissions contain the actual
`NativeInputUseAsset` handle and all four origin keys.

```python
from rexgraph_cpp import PreparedAvailableFiniteFieldMap

model = PreparedAvailableFiniteFieldMap.from_archive(map_bytes, execution_context)
family = model.execute(execution_context)
branches = family.select(0, 1, execution_context)
branch = branches[0]
assert branch.family(execution_context).archive() == family.archive()
```

`finite_constraint_origin_payload` and `finite_interpretation_origin_payload`
produce complete payloads for explicit publishers. To complete an interpreted
branch, publish its `completion_payload(ordinal, free_values, parameter_setting,
execution_context)` as a separate Prospective origin, then call
`evaluate(ordinal, free_values, parameter_setting, choice_origin, execution_context)`.
Affine branches require a positive independently published global parameter setting,
including when the complete free vector is empty. Fixed branches use setting key
zero and an empty free vector. Selected sections, branches and evaluations retain
all alternatives in their whole parent family.

Archives are passive and grant no execution authority. Available inspection and
selection require the current context. `raw()` explicitly exposes an unadmitted
mathematical owner. Finite table training and promotion of finite results to local right hand sides are
unsupported.

## Reusing a finite operator on another input

A stored finite map retains its original rule inputs. To evaluate another input,
declare an operator with ports and a source/use transport, then supply a complete
field at each port. The transport carries the relationship to the reference
source; equal numerical encodings cannot establish that relationship.

Raw reusable finite operators retain their complete reference map and execute
independent whole input fields through a declared bijective source/use transport.
`PreparedFiniteFieldOperator(reference_map, declaration)` requires one distinct
port for every original numerical input, including inputs of unmatched rules.

```python
from rexgraph_cpp import PreparedFiniteFieldOperator, NativeFiniteSource

operator = PreparedFiniteFieldOperator(reference_map, operator_declaration)
input_source = NativeFiniteSource(
    independent_asset, operator.source_declaration(independent_asset, transport))
schema = operator.input_schema(independent_asset, transport, port_key)
instance = operator.instantiate(input_source, instance_declaration, [
    {"port": port_key, "schema": schema, "use": input_use,
     "field": whole_input_field_archive},
])
family = instance.execute()
branches = family.select(0, family.inspect()["branch_count"])
choice = family.evaluate(selected_ordinal, complete_free_values)
whole_response = choice.raw().result()
```

Declare all ports and submit all their input fields when the reference map has
multiple inputs. Source declarations and `NativeFiniteInputSchema` are passive
native builders; schema inspection returns exact matrix archives and retained
stalk metadata. Instance `select_inputs`, transport `variable_positions` and
`use_correspondences` take explicit bounded ranges. Operator, instance, family
and choice archives replay their complete ancestry. Raw values require a separate
admitted execution contract for common model/service operations. Keep the runtime
recorded by a pinned model; a different native build rejects its wrapper.
Archive replay must admit every ancestor's retained resource policy. Pass explicit
`limits` when the producing native policy exceeds the Python defaults.

### Publishing independent operator inputs

`PreparedAvailableFiniteFieldOperator` retains the admitted reference map and
requires explicit Maps and relationship origins. Its passive `source_declaration`
and `input_schema` builders prepare independent inputs before publication.
`finite_operator_input_origin_payloads` returns the exact four input role payloads;
`transport_payload`, `child_map_payload` and `instance_payload` bind the transport,
child map and positive instance lineage. Publish the instance payload separately
before calling `instantiate`. The payload builder uses the instance key without
requiring its own receipt.

```python
family = admitted_instance.execute(execution_context)
branches = family.select(0, 1, execution_context)
choice_bytes = family.completion_payload(
    ordinal, complete_free_values, parameter_setting, execution_context)
# Publish choice_bytes as the separate Prospective choice_origin.
choice = family.evaluate(
    ordinal, complete_free_values, parameter_setting, choice_origin,
    execution_context)
whole_response = choice.raw().raw().result()
```

All admitted inspection, selections, dependency views, replay and actions require
the current independent context. The full original reference and target/evidence
closure, all independent input roles including unmatched ports, transport and
child publications and instance lineage are rechecked before execution. A global
affine setting remains separate from the input specific operator choice, including
when the affine free vector is empty. Raw getters are passive mathematical escapes;
`field_map` exposes the separately admitted child with its declared narrower scope.
The four operator/instance/family/evaluation archives retain complete ancestry.

## Native source revisions

`NativeSourceEvolution` owns a complete native temporal series. Its `append(frame)`
and `add_declarations(specification)` methods return new owners while retaining
the parent's bytes and earlier endpoints. Resource limits remain attached to each
fork and apply to its resulting history as well as its input.

```python
from rexgraph_cpp import NativeSource, NativeSourceEvolution

source = NativeSource.from_declaration({
    "profile": "declared-exact-source-v1", "namespace": 42,
    "semantic_scale": 1, "max_grade": 1,
    "types": (), "witnesses": (10,), "relations": (),
    "carriers": (), "fields": (), "metrics": (),
})
parent = NativeSourceEvolution(source.archive)
child = parent.add_declarations({
    "profile": "source-signature-additions-v1", "aspects": (),
    "types": ({"name": "observation", "min_arity": 0, "max_arity": 4},),
})
assert parent.identity == (42, 0, 0, 0)
assert child.identity == (42, 0, 1, 1)
assert child.source().at(0).archive == parent.source().at(0).archive
assert NativeSourceEvolution(child.archive).archive == child.archive
```

An aspect addition supplies `name`, `role`, `builder_default`, `fill` and ordered
`assignments`. A boolean fill explicitly covers existing occurrences; with
`fill=None`, assignments must cover every occurrence exactly once, including
inactive and supported-zero occurrences. Each assignment is a
`(grade, occurrence_key, boolean)` tuple. Additions preserve signed incidence
aspects, all original fields and support. New relation types use explicit minimum
and maximum arity with declared exact boundaries.

`append` takes a canonical native temporal frame for the selected parent.
Invalid frames, wrong parents and resource excesses reject without changing that
parent. The owning `source()` view can outlive earlier Python owners. These
operations do not infer a map from a Python graph or prove declared ontology axioms.
The [Core ontology guide](../../rexgraph/README.md#ontology-snapshots) and
[query guide](../../rcql/README.md#ontology-queries) describe the enclosing values.
