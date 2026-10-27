# RexGraph Agent

Agent brings data into RexGraph, runs analysis and connects the result to
application workers. A reader constructs the relationships, Core computes on
them, and RCDB stores the selected state. Agent supplies the server, pipeline,
connectors and worker lifecycle around those operations.

A worker can analyze, transform, score or generate; it can be a Python callable
or a service with an OpenAI compatible endpoint. A language model is one kind
of worker. The analysis and callable examples below run without model weights.

## Installation

From the repository root:

```sh
make install
make serve
```

The equivalent script is `sh install.sh`. Both install the standard profile
and all five repository packages in `.venv`. Open
[http://127.0.0.1:8000](http://127.0.0.1:8000) and use the admin token printed
on first startup. The HTTP API documentation is at `/docs`.

| Command | Agent dependencies |
| --- | --- |
| `make install-server` | Web application and API |
| `make install` | Server, schema tools, connectors, PDF/OCR bindings, training export and YAML |
| `make install-ml` | Standard profile plus PyTorch and Hugging Face tools |
| `make install-dev` | Standard profile and test tooling, installed as editable packages |
| `make install-all` | Declared runtime extras, including cloud and model integrations |
| `make install-dev-full` | Development dependencies for optional CPU feature tests |

Add an integration with `sh install.sh --extras trustgraph`, or combine extras
as `--extras langchain,langgraph`. Install Tesseract for scanned documents,
`llama-server` for managed local inference, and a compatible PyTorch build for
GPU training. The [installation guide](../README.md#installation) covers these
optional runtimes.

## SQL sources and native imports

Use RCDB for Agent records and RCQL programs. SQL databases and warehouses remain
optional read sources. Import their data into RCDB to keep a verified local copy,
retain source declarations and run queries without the original SQL service.

Install SQL support with `sh install.sh --extras sql`. Add the SQLAlchemy driver
for your database; `--extras warehouse` includes Snowflake, BigQuery and DuckDB
adapters. These dependencies are optional and do not change the default RCDB store.

```sh
rexgraph-connect import-sql --source-env DATABASE_URL \
  --store local:///data/records --id warehouse \
  --files /data/native-tables --format rcbd
```

`DATABASE_URL` contains the SQLAlchemy connection URI. An explicit URI or saved
connection name can also be passed as the positional source. The command imports
all tables in the selected schema; use `--schema NAME` and repeated `--table NAME`
options to select a subset. `--batch-size` bounds each row batch. To publish another
revision of an existing import, supply its `--expected-version`.

In **RCQL Records**, select a saved SQL source, enter a source URI, or upload a
SQLite file. **Inspect SQL source** shows its tables and keys. **Import all tables
into RCDB** imports and verifies the data. **Download native files** exports the
import as RCBF, RCBD, safetensors or HDF5 files. Downloads contain a ZIP of the
manifest and table batches. Use `--format rcbf` for sequential binary files or
`--format rcbd` for addressable directory bundles. RCBF output uses version 6;
older readers that accept only versions 4 and 5 must be updated to read it.

Imports retain row occurrences, column order, nulls, driver scalar types, declared
primary and foreign keys, and available indexes/checks. They do not infer missing
relations or execute SQL defaults, triggers, procedures or views. Each import records
its source transaction isolation; repeatable snapshot guarantees depend on that source.
SQL reads use explicit table and column selectors rather than generated SQL text.

```python
from agent.agentic_db import AgenticDB
from agent.sql_source import SQLDataset

with AgenticDB("local:///data/records") as db:
    dataset = db.sql_dataset("warehouse")
    for row in dataset.rows("orders"):
        print(row)

# Native exports need neither SQLAlchemy nor the original SQL service to read.
dataset = SQLDataset.from_directory("/data/native-tables")
```

The source database is read without modification. Failed imports retain any
published batches for inspection but do not publish a complete dataset manifest.
Native exports are checked against stored identities before publication. General
SQL table rows are retained as explicit native row assertions; schema diagnostics
and application-specific relational models are separate views of that data.

## Analysis

```python
from rexgraph import RexGraph
from agent.pipeline import AnalysisPipeline

rex = RexGraph.from_graph([0, 1, 2], [1, 2, 0])
result = AnalysisPipeline(rex).run(depth="standard")
print(result["hodge"]["dim_H"])
```

The pipeline reports Hodge components and structural channel measurements.
Hodge decomposition separates gradient flow from vertex potentials, curl flow
supported by face boundaries, and harmonic flow in the Hodge operator's kernel.
For this unfilled cycle the harmonic dimension is one. A different adapter can
construct different relations from the same file, so inspect that construction
before assigning an application meaning to the measurements.

`agent.auto.auto_rex` accepts edge tables, JSON, NumPy arrays, DataFrames and
text. The selected adapter determines which relationships are constructed from the
input. Core's [declared readers](../rexgraph/io/README.md) let an application
specify that mapping, source types, exact number conversion and provenance.

## Data and optional features

The standard profile includes common file readers, PDF extraction and Python
OCR bindings. Scanned documents also require the Tesseract executable. Model
weights and database credentials are configured for the selected application.
`make doctor` reports optional packages and external tools; `make verify`
checks the installed software.

PDF rendering rejects incomplete page sets and stops its worker if conversion
exceeds five minutes. Temporary images are removed after processing or failure.
Model discovery checks configuration files and every indexed weight shard before
reporting a download as available. Model loaders validate file contents and the
required tokenizer or processor files when loading the model.

| Extra | Use |
| --- | --- |
| `molecules` | RDKit input and molecular interpretation |
| `analytics` | DuckDB and Polars views of stored signatures |
| `baselines` | scikit learn, LightGBM and XGBoost comparisons |
| `ml` | PyTorch and Hugging Face model tools |

Install additional dependencies with `sh install.sh --extras analytics,molecules`
from the repository root. The [dependency guide](../DEPENDENCIES.md) lists the
other extras and their requirements.

## Configuration and cache

Choose directories before starting Agent or its command line tools. For example:

```sh
export REXGRAPH_CONFIG_DIR=/path/to/settings
export REXGRAPH_CACHE_DIR=/path/to/cache
export REXGRAPH_MODELS_DIR=/path/to/models
export REXGRAPH_SESSION_DIR=/path/to/sessions
make serve
```

| Setting | Purpose |
| --- | --- |
| `REXGRAPH_CONFIG_DIR` | Configuration, credentials, workspace records and journals |
| `REXGRAPH_CACHE_DIR` | Reusable analysis results and OCR server process files |
| `REXGRAPH_MODELS_DIR` | Explicit model downloads; defaults to `models` under the cache directory |
| `REXGRAPH_MODEL_SCAN_DIRS` | Local model discovery roots, separated by the platform path separator |
| `REXGRAPH_SESSION_DIR` | Saved analysis sessions |
| `REXGRAPH_NO_CACHE=1` | Disable reusable analysis and JSON result caching |
| `REXGRAPH_CACHE_MAX_ENTRIES` | Entries retained after pruning; defaults to 2000, with 0 disabling result caching |

Without overrides, Linux and macOS use `XDG_CONFIG_HOME` and `XDG_CACHE_HOME`,
falling back to `~/.config/rexgraph` and `~/.cache/rexgraph`. Windows uses
`APPDATA` and `LOCALAPPDATA`. Sessions default to `sessions` under the
configuration directory. Passing `storage_dir` to the session API selects
a directory for that session.

Model discovery checks the configured model directory. Set
`REXGRAPH_MODEL_SCAN_DIRS` to choose discovery roots. Additional directories
supplied through `REXGRAPH_MODEL_DIRS` or an explicit discovery call are included.

Session writes publish graph data before updating the history index. A failed
write raises an error and retains the previous history. A session loaded before
another writer appends history must be reloaded before it can write. Copied or
moved session directories retain access to their contained snapshots.

File, OCR and pipeline uploads use `REXGRAPH_MAX_UPLOAD_MB`, which defaults to
100 MiB. The limit covers the total file content in a pipeline request.
Invalid or nonpositive values use the default. Uploaded temporary files are
removed after processing, failed reads or cancellation.

Explicit RexGraph settings take precedence over platform defaults and support
`~` in paths. The `cache_dir` and `models_dir` fields in `config.json` also
select directories; environment overrides take precedence over saved values.
Hugging Face downloads use its separate cache settings,
including `HF_HOME` and `HF_HUB_CACHE`; model discovery reads those locations.

Analysis cache keys include input content, analysis parameters and the cache
format version. Stored complex data and its sidecar must have matching digests.
A missing, stale or damaged entry is recomputed. Pruning retains the most recently
written entries and runs on the first successful write in each process, then
every 128 writes. The count can temporarily exceed the configured limit between
sweeps. Pruning preserves model files and service configuration.

Chat reuses deterministic single document answers only when document state and
retrieval parameters match. Live model generation and queries
over mutable corpora or stores bypass the response cache.

## Query records and execute programs

Agent installs RCDB and RCQL as runtime dependencies. `RelationalSession` uses the
same configured default store as the server. Pass an explicit store for an
embedded application:

```python
from agent import RelationalSession
from rexgraph import RexGraph

session = RelationalSession()
session.publish("example-data", RexGraph.from_cells([3, [[0, 1], [1, 2]]]),
                expected_version=0)
result = session.query(
    'FROM RCDB_VERSION($db,"example-data",1) RETURN COUNT(CELLS(1)), SUM([1/3,2/7])',
    save_as="example-answer")
print(result.values)  # (2, Fraction(13, 21))
```

An expected version of 0 creates a new record. Use its current version to append
a revision; a stale value refuses. The example ids must be unused on its first
run. Queries are read-only by default. A mutation requires `allow_write=True`
and an explicit `expected_version` in the query. Training also requires the
write permission; source/provider availability rules still apply.

Publish and execute finite programs through the same session:

```python
from fractions import Fraction

session.publish_program("example-double", {
    "name": "Double",
    "inputs": [{"name": "x", "kind": "ExactRational"}],
    "steps": [{"name": "twice", "query": "FROM $data RETURN SUM([$x,$x])"}],
}, expected_version=0)
result = session.run_program("example-double", version=1,
    parameters={"x": Fraction(2, 3)},
    sources={"data": {"record_id": "example-data", "version": 1}})
print(result.values)  # (Fraction(4, 3),)
```

`sources` binds stored values under explicit aliases; omitted versions select
the current version and execution records the version actually selected. `db`
always names the session's governed store. `record_parameters` loads typed input
records; its `kind` can be `value`, `program`, `name` or `recursive`. An optional
`output` index selects a value from a saved canonical Result before decoding.
The selected record/version/output identity remains in execution provenance. Python
callers can also supply native values directly in `parameters`. Text is parsed
by the session entry point, while the lower `RCQLRuntime.execute` continues to
accept typed queries only.

The publication tool can also use `value_record` with the same selector. For
example, a query returning `NAME_RECORD(object)` can save its result, and
`rexgraph_rcdb_publish` can publish output 0 as the named object's record. The
record retains its parent result id, version, output and state digest. Subsequent
queries restore that object through `record_parameters` without copying its
coefficients into JSON. Repeating a stored operation uses the existing
`NAME_ITERATE` or bounded recursive constructs under the same source policies.

`transform_program` accepts the existing `specialize`, `input` and `step` rules,
checks the candidate against supplied sources and validation parameters, and
publishes a guarded revision. Its canonical transformation certificate is stored
in that same program record. Historical versions remain executable.

The shared capability registry exposes these operations directly to tools,
Hive workers and remote clients:

| Tool | Operation |
| --- | --- |
| `rexgraph_rcdb_list` | List visible records |
| `rexgraph_rcdb_inspect` | Inspect a selected version and optional history |
| `rexgraph_rcdb_publish` | Publish Core cells or a selected stored value with a version guard |
| `rexgraph_rcql_query` | Execute a query or explicitly permitted mutation |
| `rexgraph_rcql_publish_program` | Publish a finite program declaration |
| `rexgraph_rcql_run_program` | Execute a selected program version |
| `rexgraph_rcql_transform_program` | Validate and publish a certified program revision |

Hive can own the execution sequence. Register capability workers and declare
their dependencies with `WorkflowStep` and `WorkflowRef`:

```python
from agent.hive import Hive
from agent.hive_workflow import WorkflowRef, WorkflowStep

hive = Hive("query-team")
hive.add_tools(names=["rexgraph_rcdb_inspect", "rexgraph_rcql_query"])
workflow = hive.run_workflow([
    WorkflowStep("inspect", "rexgraph_rcdb_inspect",
                 {"record_id": "example-data", "version": 1}),
    WorkflowStep("query", "rexgraph_rcql_query", {
        "query": "FROM $data RETURN COUNT(CELLS(1))",
        "sources": {"data": {
            "record_id": WorkflowRef("inspect", ("record", "id")),
            "version": WorkflowRef("inspect", ("record", "version")),
        }},
    }),
])
print(workflow.outputs["query"]["values"])
```

The coordinator schedules ready workers. References pass actual whole values
or explicit mapping/sequence selections. Register a bounded model callback with
`add_worker` to provide an input, choose an admitted operation or interpret a
result; the model does not need to dispatch tools. A separate application worker
can turn that checked decision into arguments for a registered query worker.
Unknown workers and dependency cycles refuse before execution. Worker failures
remain in `workflow.steps` and block dependent steps; independent work can finish.
Scheduler failures do not automatically repeat mutations. Completed record writes
remain committed: the workflow does not provide multi-record rollback. Outputs
are in-process values; existing RCQL/RCDB codecs persist selected complete results.

Register any selected operations using `hive.add_tools(names=[...], context=...)`.
The existing `/rex/v1/call` and `/api/v1/mcp/call` routes invoke the same handlers.
`RexClient` provides `rcdb_list`, `rcdb_inspect`, `rcdb_publish`, `rcql_query`,
`rcql_publish_program`, `rcql_run_program` and `rcql_transform_program` through
that existing transport. For example:

```python
from agent.client import RexClient

client = RexClient("http://localhost:8000", api_key="TOKEN", workspace="default")
reply = client.rcql_query(
    'FROM RCDB_VERSION($db,"example-data",1) RETURN COUNT(CELLS(1))')
print(reply["result"]["values"])
```

Tool parameters use JSON; exact fractions use
`{"numerator": 2, "denominator": 3}`. Responses include a bounded `values`
preview and, by default, `result_base64`. Decode the complete payload with
`rcql.Result.from_bytes(base64.b64decode(payload))`. Native fields and object
arrays are not expanded for the preview. A program result also retains complete
intermediary values in `result.native_plan["step_results"]`, the program
declaration, source dependencies, arithmetic and execution observations.

`save_as` publishes the canonical result using RCDB's declared query-result
codec. Native Memory, Local and SQLite stores support it. Legacy stores and
stores whose immutable header excludes that codec refuse; they are not silently
rewritten. Query execution and saving its result are separate operations. Each
record publication uses the existing engine's version guard; the workflow does
not provide a multi-record atomic transaction.

Authenticated calls and context-bound Hive tools use the existing workspace
store view, actor stamping, audit trail and cell admission limits. A caller
cannot supply a store URI or arbitrary filesystem source through these tools.
Outside a socket or explicit context, `RelationalSession` has the same trusted
local-operator scope as the rest of Agent. Native model execution still requires
its owning provider and runtime; this interface does not change model loaders,
source cutoffs or transport contracts.

## The Hive

`Hive.add_worker` registers a callable with a declared capability. Supported
capabilities are `generate`, `predict`, `score`, `embed`, `analyze` and `transform`.
Use `invoke` to call a named worker and `dispatch` to route a text request to a
worker with the `generate` capability.

```python
from agent.hive import Hive
from rexgraph import RexGraph

hive = Hive()

def analyze(data, **kwargs):
    graph = RexGraph.from_graph(data["sources"], data["targets"])
    return {"betti": tuple(graph.betti)}

hive.add_worker("structure", analyze, capability="analyze", specialties=["homology"])
answer = hive.invoke("structure", {"sources": [0, 1, 2], "targets": [1, 2, 0]})
print(answer)
```

`attach` connects to an existing OpenAI compatible endpoint. `spawn` starts a
managed llama.cpp server for a model file. `rexgraph-bee` exposes the same
endpoint contract to a responder that polls for requests. These use two
transports: an in process callable or an HTTP endpoint.

```python
hive.attach("chat", "http://127.0.0.1:8080", role="worker", specialties=["homology"])
reply = hive.dispatch("Explain the first Betti number of a cycle.")
```

The hive records interactions in an activity complex and uses declared
specialties and interaction history for routing. `hive.status()` returns
membership and health. `get_hive()` supplies the process singleton; create
separate `Hive` instances when tool registrations must be independent.

## Local inference

Install or build llama.cpp separately and make `llama-server` available on
`PATH`, or configure an existing binary through the local runtime. The RexGraph
installer compiles RexGraph's Core extensions.

```sh
.venv/bin/rexgraph-local --help
.venv/bin/rexgraph-hive --help
```

A managed server uses a local GGUF model file. An attached service can use
llama.cpp, vLLM or another implementation of the endpoint contract. Model files,
GPU drivers and external inference runtimes have their own installation steps.

## Models and connectors

The [model builder](agent/models/README.md) assembles MLP, CNN, language and
hypergraph models from `rexgraph.nn` components. Core's
[cochain models](../rexgraph/flow/README.md) use cells as their parameter space.
Feature models and cochain models use different optimizers and checkpoint
formats, as described in their respective guides.

[Connectors](agent/connectors/README.md) read declared source structure and
construct a relational complex. Optional live drivers support SQL, MongoDB,
Neo4j and cloud warehouses. Inspect the available schemes before configuring a
source:

```sh
.venv/bin/rexgraph-connect list
```

## Lifecycle and command line tools

A setup is a named configuration for paths, models, backend and thread settings.
Lifecycle operations use that setup and record their runs.

| Command | Purpose |
| --- | --- |
| `rexgraph-config` | Inspect and manage setups |
| `rexgraph-ops` | Serve, train, ingest, build, deploy, test and benchmark |
| `rexgraph-run` | Run the document analysis pipeline |
| `rexgraph-models` | Build and train model archetypes |
| `rexgraph-connect` | List, validate and ingest data sources |
| `rexgraph-deploy` | Write a container deployment bundle |
| `rexgraph-auth` | Manage authentication and workspace members |
| `rexgraph-courier` | Transfer selected records between stores or peers |

These executables are in `.venv/bin`. Use `--help` for their arguments.
`make run FILES='document.pdf'` runs the pipeline from the checkout.
[Benchmarks](agent/benchmarks/README.md) document measurements and their limits.

## Integrations

Optional extras install each integration's dependencies:

| Extra | Interface |
| --- | --- |
| `trustgraph` | Knowledge graph imports, structural retrieval measurements and enrichment triples |
| `huggingface` | Transformer attention analysis |
| `langchain` | Structural analysis tools for LangChain agents |
| `langgraph` | State graph construction and analysis |
| `warehouse` | SQLAlchemy dialects for Snowflake, BigQuery, Redshift and Databricks |
| `objectstore-cloud` | S3, Google Cloud Storage and Azure filesystem drivers |
| `oidc` | OIDC authentication support |

Standalone triple analysis needs no TrustGraph service:

```python
from agent.integrations.trustgraph_adapter import TrustGraphAdapter, SimpleTriple

adapter = TrustGraphAdapter()
rex, metadata = adapter.from_triples([
    SimpleTriple("compound", "targets", "protein"),
    SimpleTriple("protein", "participates_in", "process"),
])
```

## Server and API

`make serve` starts the Agent application. `rcf-server` is its installed console
entry point. `rexgraph-serve` starts the separate model/OCR inference service.

Analysis and session routes use `/api`. Application services use `/api/v1`.
Native artifact and record transfer uses `/rex/v1` through `agent.client.RexClient`.
The server includes its UI assets and licensed React runtime for offline use.

## Authentication

A fresh server enables authentication and prints a bootstrap admin token once.
Members receive bearer tokens with roles in named workspaces. Workspace admins
manage members and operations requiring an admin role; users have the permissions granted
to their workspace role. The activity journal records the acting identity.

```sh
.venv/bin/rexgraph-auth network-init
.venv/bin/rexgraph-auth member add --name alice --role user
.venv/bin/rexgraph-auth member list --workspace default
```

For a shared deployment, configure TLS with `REXGRAPH_TLS_CERT` and
`REXGRAPH_TLS_KEY`, or use a trusted reverse proxy. `rexgraph-deploy` writes
Compose configuration, environment settings and volume declarations. Copy its
`.env.example` to `.env` and configure the deployment before starting it.

## Deployment

The default application binds to `127.0.0.1:8000`. Configure `RCF_HOST` and
`RCF_PORT` for another bind address or port. Rate limits apply per server worker;
a proxy can enforce limits across several workers.

`make serve-https` generates a local certificate when needed. Use supplied
certificate and key files for a deployment with a trusted certificate.
`make serve-dev` enables source reload for an editable development installation.

The compute layer selects CPU, CUDA, ROCm or MPS according to the requested
backend and available runtime. An explicit GPU request raises an error when
the device or required runtime is unavailable.

## External readers

Dataset adapters, source clients and ontology mappings belong in external
packages. A reader can return `agent.adapters.EdgeConstruction`, retaining
primary relation supports, source identity and structured cell attributes.
Use `agent.auto.build_rex_from_edges` to construct the native object.

For file dispatch, register the external reader explicitly:

```python
from agent.adapters.formats import register_reader, read, unregister_reader
from my_readers import load_records

register_reader("my_records", load_records, extensions=(".records",))
try:
    construction = read("input.records", source_release=selected_release)
finally:
    unregister_reader("my_records")
```

Register dataset readers before using them. The reader interprets its source
data, Core constructs the complex, RCQL queries it and RCDB stores it.

## Stored record transfer

RCDB record packets retain a selected payload version, metadata, validity and
source identity. `agent.client.RexClient` transfers those packets through
`/rex/v1`, and couriers use the same `rcdb.copy_record` contract for local and
remote destinations. Destination workspace and integrity policies apply at
publication. A sender ledger retains acknowledgements and supports explicit
reconciliation when a checked receipt is available.

See the [record transfer guide](RECORD_TRANSFER.md) for version selection,
concurrency, uncertain outcomes and ledger retention, and the
[RCDB guide](../rcdb/README.md#portable-record-versions) for the packet contract.

## License

Agent is distributed under the [Apache License 2.0](LICENSE).

## Ontology runtime

Agent's registered `.rcbo` reader and `auto_rex` load the whole relational source
with its ontology declaration. `agent.ontology_runtime.read_ontology` uses the
existing `RelationalSession`, checks the selected record's actual publication
time as well as its declared knowledge horizon, and resolves evidence pins
under the requested cutoff. The cutoff must be an `ExactTime`.

`publish_ontology` publishes through the same session with an explicit expected
record version. A child must identify the actual published parent. RCQL programs
and Hive workers carry whole ontology values through the existing program,
source-authority and record interfaces. See the
[ontology query guide](../rcql/README.md#ontology-queries).

`native_revision_program(name, operation="append")` declares a revision program
with component, frame bytes and exact integer cutoff inputs. With `operation="add"`,
the specification input carries explicit native aspect/type additions instead.
Register the stored program through `Hive.add_program` with an explicit parent
record version, source selectors and `allow_write=True`. The default permits
read operations only; invocation arguments cannot override this permission.
Execution returns a child ontology. Publish it separately with
`publish_ontology(..., expected_version=parent_version)`, which checks the actual
parent again. Native source integrity and declared axiom validation remain
separate checks.

Agent record publication accepts RCDB file, memory and admitted object-store
providers. SQL sources are read through the SQL source/connector APIs; standalone
RCDB SQL storage remains available outside Agent publication.
