"""Complete native relational runtime snapshots in ordinary RexGraph records."""
from hashlib import sha256

import numpy as np


def relational_record(runtime):
    from rexgraph.graph import RexGraph
    from rexgraph.relational_runtime import RelationalFieldFunction, RelationalRuntime
    if not isinstance(runtime, (RelationalRuntime, RelationalFieldFunction)):
        raise TypeError("relational record requires an owning native runtime or field function")
    raw = runtime.to_bytes()
    record = RexGraph.from_hypergraph([0, 1], [0])
    record.attach_metadata(1, 0, "relational_runtime", {
        "profile": "rexgraph.relational-runtime-record.v1", "sha256": sha256(raw).hexdigest(),
        "packet": np.frombuffer(raw, np.uint8).copy()})
    return record


def read_relational_record(record, index=0, *, restore_limits=None):
    from collections.abc import Mapping
    from numbers import Integral

    from rexgraph.relational_runtime import RelationalFieldFunction, RelationalRuntime
    from rexgraph.value_codec import unpack_value
    if isinstance(index, bool) or not isinstance(index, Integral) or index < 0:
        raise ValueError("relational record index requires a nonnegative integer")
    data = record.get_metadata(1, int(index), "relational_runtime")
    if (not isinstance(data, Mapping) or set(data) != {"profile", "sha256", "packet"}
            or data["profile"] != "rexgraph.relational-runtime-record.v1"):
        raise ValueError("selected record has no complete relational runtime attachment")
    packet = data["packet"]
    if not isinstance(packet, np.ndarray) or packet.dtype != np.uint8 or packet.ndim != 1:
        raise ValueError("relational runtime attachment requires a byte vector")
    if packet.size > 132 << 20:
        raise ValueError("relational runtime attachment exceeds admission")
    raw = packet.tobytes()
    if sha256(raw).hexdigest() != data["sha256"]:
        raise ValueError("relational runtime attachment digest differs")
    data = unpack_value(raw)
    if type(data) is dict and data.get("profile") == "rexgraph.relational-field-function.v1":
        return RelationalFieldFunction.from_bytes(raw)
    options = {} if restore_limits is None else {"restore_limits": restore_limits}
    return RelationalRuntime.from_bytes(raw, **options)
