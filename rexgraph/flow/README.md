# RexGraph cochain models

A label observed on one relation can inform nearby relations that share its
participants. These PyTorch models put trainable values on the cells of a
complex and use a Green operator to propagate parameter updates through that
structure. The example below keeps a separate mask for observed labels, so
propagation can update a row whose label did not enter the loss.

This is the numerical cochain model family included in Core. Exact native
whole field programs use a separate [model interface](../README.md#exact-native-models)
with source, map and alternative family declarations. Choose the model family
before choosing an optimizer or checkpoint format.

## Installation

From the repository root:

```sh
make install-ml
```

For an existing Core environment, install `.[nn]` from the repository
root. That extra supplies PyTorch and SciPy for the numerical model components.
See the [installation guide](../../README.md#optional-runtimes) for GPU builds.

Sparse operators use the parameter dtype and device, and safetensors reload
preserves the stored dtype. The online field refreshes cached operators after
graph edits. Retained model records identify their source, coordinates and
implementation; Agent feature model checkpoints use the
[warm start behavior](../../agent/agent/models/README.md#checkpoint-semantics)
described in their guide.

## Models

| Component | Purpose |
| --- | --- |
| `CoParticipationCochain` | Class parameters with one row per relation, trained from observed labels |
| `CoParticipationAttention` | Learned attention over the coparticipation adjacency |
| `MalaughGate` | Detect changes in a structural entropy measurement |
| `FieldNavigator` | Apply Hodge flow on a selected region |
| `GreensCochainField` | Predict and update an online field as observations arrive |

`CoParticipationCochain` exposes `greens_groups()`, which lets the automatic
optimizer select `GreensCochain`. It combines Adam updates with Green operator
propagation through coparticipation. Relations connected to observed labels can
therefore receive parameter updates even when their own rows contribute no
gradient to the loss.

`GreensCochainField` updates as observations arrive, using a Green solve and
relational correction. Its state, caching and update options are defined in
[online.py](online.py).

## Usage

Represent each measured relationship as a relation and each participating entity
as a vertex. Relations sharing an entity become coparticipants. A relation with
more than two participants uses `RexGraph.from_hypergraph` or declared `Relations`.
A high degree vertex and a relation with several participants are distinct.

```python
import numpy as np
from rexgraph import RexGraph
from rexgraph.flow.cochain import CoParticipationCochain

rex = RexGraph.from_graph([0, 0, 1, 2], [1, 2, 3, 3])
labels = np.array([0, 0, 1, 1], dtype=np.int64)
observed = np.array([True, False, True, False])
model = CoParticipationCochain(rex, n_classes=2)
model.fit(labels, observed, epochs=30, lr=0.1)
predictions = model.predict()
```

Only observed edges contribute to the loss. `GreensCochain` propagates their
updates through the coparticipation operator to connected parameter rows.

## Optimizer selection

Build the optimizer with `make_optimizer("auto", model, params)`. It selects
`GreensCochain` for models exposing `greens_groups()` and Adam for feature models.
Use `RexGraph` sparse properties for structure and the dense kernels for explicit
reference comparisons. Core also supplies seeded response and similarity methods:
`coherence_response(seed)`, `character_response(seed)`, `local_context(seed)`,
`phi_similarity_score`, `chi_cosine` and `similarity_complex`.
