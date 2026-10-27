"""Complete action packets in ordinary RexGraph application records."""
from collections.abc import Mapping
from hashlib import sha256

import numpy as np

_MAX_BYTES = 132 << 20


def action_record(value):
    from rexgraph.action_program import ActionEvidence, ActionOutput, ActionProgram
    from rexgraph.graph import RexGraph
    if not isinstance(value, (ActionEvidence, ActionOutput, ActionProgram)):
        raise TypeError("an owning native action program, evidence request or output is required")
    packet = value.to_bytes()
    record = RexGraph.from_hypergraph([0, 1], [0])
    record.attach_metadata(1, 0, "native_action", {"profile": "rexgraph.action-record.v1",
        "sha256": sha256(packet).hexdigest(), "packet": np.frombuffer(packet, np.uint8).copy()})
    return record


def action_packet(record):
    data = record.get_metadata(1, 0, "native_action")
    if (not isinstance(data, Mapping) or set(data) != {"profile", "sha256", "packet"}
            or data["profile"] != "rexgraph.action-record.v1"):
        raise ValueError("selected record has no complete action attachment")
    vector = data["packet"]
    if (not isinstance(vector, np.ndarray) or vector.dtype != np.uint8 or vector.ndim != 1
            or vector.size > _MAX_BYTES):
        raise ValueError("bounded action byte vector required")
    packet = vector.tobytes()
    if sha256(packet).hexdigest() != data["sha256"]:
        raise ValueError("action record packet digest differs")
    return packet


def read_action_record(record, *, evidence=None, availability=None):
    from rexgraph.action_program import ActionEvidence, ActionOutput, ActionProgram
    from rexgraph.value_codec import unpack_value
    packet = action_packet(record)
    data = unpack_value(packet)
    if type(data) is not dict:
        raise ValueError("unsupported action attachment")
    profile = data.get("profile")
    if profile == "rexgraph.action-evidence.v1":
        return ActionEvidence.from_bytes(packet)
    if availability is None:
        raise ValueError("current native publication availability required for action replay")
    if profile == "rexgraph.action-program.v1":
        return ActionProgram.from_bytes(packet, availability=availability)
    if profile == "rexgraph.action-output.v1":
        return ActionOutput.replay(packet, evidence=evidence, availability=availability)
    raise ValueError("unsupported action attachment")
