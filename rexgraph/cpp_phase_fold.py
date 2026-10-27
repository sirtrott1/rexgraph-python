"""Exact native phase/fold words, prefix observations and complete source records."""

from collections.abc import Mapping
from copy import deepcopy
from fractions import Fraction
from hashlib import sha256
from math import prod

import numpy as np

from rexgraph.cpp_algebraic import _key, _limit_names
from rexgraph.source_binding import CppSourceReference, _sha, source_from_record
from rexgraph.value_codec import pack_value

__all__ = [
    "phase_fold_reference",
    "validate_phase_fold_reference",
    "phase_fold_record",
    "read_phase_fold_record",
    "phase_fold_observation",
    "validate_phase_fold_observation",
    "phase_fold_observation_record",
    "read_phase_fold_observation_record",
    "phase_fold_history_record",
    "phase_fold_history_at",
]

LIMIT_NAMES = frozenset("wire." + name for name in _limit_names) | {
    "history.max_steps",
    "history.max_coordinate_slots",
    "history.max_integer_bits",
    "history.max_source_cells",
    "history.max_source_occurrences",
    "history.max_interaction_terms",
    "cover.max_vertices",
    "cover.max_path_steps",
    "cover.max_integer_bits",
    "panels.max_cells",
    "panels.max_occurrences",
    "panels.max_creases",
    "panels.max_integer_bits",
    "max_proposals",
}
KINDS = frozenset(
    {"rational-phase", "algebraic-phase", "rational-fold", "rational-walk", "algebraic-walk"}
)
MANIFEST_NAMES = frozenset(
    {
        "codec",
        "archive_digest",
        "source_digest",
        "word_kind",
        "operation_count",
        "proposal_count",
        "field_degree",
        "source_sizes",
        "cell",
        "minimal_polynomial",
        "involution",
        "endpoint_shape",
        "endpoint_coordinate_count",
        "observation_coordinate_count",
        "branch_vertex",
    }
)


def _closed(value, names):
    if not isinstance(value, Mapping) or set(value) != set(names):
        raise ValueError("incomplete phase/fold declaration")


def _coordinates(value, size=None, *, integer_bits):
    if not isinstance(value, (tuple, list)) or (size is not None and len(value) != size):
        raise ValueError("incomplete exact phase/fold coordinates")
    if any(type(q) is not Fraction for q in value):
        raise ValueError("phase/fold coordinates require exact rationals")
    if any(
        max(abs(q.numerator).bit_length(), q.denominator.bit_length()) > integer_bits for q in value
    ):
        raise ValueError("phase/fold coordinate exceeds its integer admission")


def _history(value, manifest):
    _closed(value, {"source", "channel", "max_bytes"})
    source = source_from_record(value["source"], allow_cpp=True)
    if not isinstance(source, CppSourceReference) or not source.native.namespace:
        raise ValueError("phase/fold history requires a native publication prefix")
    names = value["channel"]
    if not isinstance(names, (tuple, list)) or len(names) != 3 or not all(_key(v) for v in names):
        raise ValueError("phase/fold history requires three nonzero channel names")
    _key(value["max_bytes"])
    if tuple(map(len, source.native.coordinates)) != tuple(manifest["source_sizes"]):
        raise ValueError("phase/fold history source coordinates differ")
    return source


def validate_phase_fold_reference(value):
    """Check the complete portable word, coefficient domain and reader policy."""
    _closed(value, {"kind", "version", "profile", "sdk", "manifest", "limits", "history"})
    if (
        value["kind"] != "cpp_phase_fold_reference"
        or type(value["version"]) is not int
        or value["version"] != 1
        or value["profile"] != "exact-phase-fold-word-v1"
    ):
        raise ValueError("unsupported phase/fold reference")
    sdk = value["sdk"]
    _closed(
        sdk,
        {"engine_version", "engine_digest", "binding_digest", "binding_version", "scalar_policy"},
    )
    if (
        sdk["engine_version"] not in ("0.2.1", "0.3.0")
        or sdk["binding_version"] != "1"
        or sdk["scalar_policy"] != "gmp-rational-exact"
    ):
        raise ValueError("unsupported phase/fold native engine profile")
    _sha(sdk["engine_digest"])
    _sha(sdk["binding_digest"])
    _closed(value["limits"], LIMIT_NAMES)
    for number in value["limits"].values():
        _key(number)
    m = value["manifest"]
    _closed(m, MANIFEST_NAMES)
    if (
        m["codec"] != "rfcxw-v1"
        or not isinstance(m["word_kind"], str)
        or m["word_kind"] not in KINDS
    ):
        raise ValueError("unsupported phase/fold word kind")
    _sha(m["archive_digest"])
    _sha(m["source_digest"])
    for name in [
        "operation_count",
        "proposal_count",
        "field_degree",
        "endpoint_coordinate_count",
        "observation_coordinate_count",
    ]:
        _key(m[name])
    if (
        not m["field_degree"]
        or not isinstance(m["source_sizes"], (tuple, list))
        or not m["source_sizes"]
    ):
        raise ValueError("phase/fold source and coefficient dimensions are absent")
    for n in m["source_sizes"]:
        _key(n)
    algebraic = m["word_kind"].startswith("algebraic")
    policy = value["limits"]
    walk = m["word_kind"].endswith("walk")
    if (
        m["operation_count"] > policy["cover.max_path_steps" if walk else "history.max_steps"]
        or m["proposal_count"] > policy["max_proposals"]
    ):
        raise ValueError("phase/fold word exceeds retained operation admission")
    if algebraic and m["field_degree"] > policy["wire.field.max_degree"]:
        raise ValueError("phase/fold field exceeds degree admission")
    for name in ["minimal_polynomial", "involution"]:
        _coordinates(m[name], integer_bits=policy["wire.field.max_integer_bits"])
    if (
        algebraic
        and (
            len(m["minimal_polynomial"]) != m["field_degree"] + 1
            or len(m["involution"]) != m["field_degree"]
        )
    ) or (not algebraic and (m["field_degree"] != 1 or m["minimal_polynomial"] or m["involution"])):
        raise ValueError("phase/fold coefficient declaration differs")
    if m["word_kind"].endswith("phase"):
        _closed(m["cell"], {"grade", "index"})
        grade, index = _key(m["cell"]["grade"]), _key(m["cell"]["index"])
        if grade >= len(m["source_sizes"]) or index >= m["source_sizes"][grade]:
            raise ValueError("phase/fold carrier cell is absent")
    elif m["cell"] is not None:
        raise ValueError("fold and walk profiles do not declare a single phase cell")
    if m["word_kind"].endswith("walk"):
        if _key(m["branch_vertex"]) >= m["source_sizes"][0]:
            raise ValueError("phase walk branch is absent")
        if m["proposal_count"]:
            raise ValueError("walk replacement profile is undeclared")
    elif m["branch_vertex"] is not None:
        raise ValueError("phase/fold word has no cyclic branch")
    shape = m["endpoint_shape"]
    if not isinstance(shape, (tuple, list)) or not shape or any(not _key(n) for n in shape):
        raise ValueError("phase/fold endpoint shape is absent")
    expected = (
        (m["source_sizes"][2], 4, 4)
        if m["word_kind"] == "rational-fold" and len(m["source_sizes"]) > 2
        else (m["field_degree"] if algebraic else 2,)
    )
    if tuple(shape) != expected or prod(shape) != m["endpoint_coordinate_count"]:
        raise ValueError("phase/fold endpoint dimensions differ")
    if m["observation_coordinate_count"] < m["endpoint_coordinate_count"]:
        raise ValueError("phase/fold observation coordinates are incomplete")
    if value["history"] is not None:
        _history(value["history"], m)
    return value


def phase_fold_reference(archive):
    """Describe a checked native word and its actual retained publication, if present."""
    from rexgraph_cpp import NativePhaseFoldArchive, build_info

    if type(archive) is not NativePhaseFoldArchive:
        raise TypeError("phase/fold reference requires an owning native archive")
    history = None
    if archive.history_owner is not None:
        history = {
            "source": CppSourceReference(archive.history_owner).as_record(),
            "channel": archive.history_channel,
            "max_bytes": archive.history_owner.max_bytes,
        }
    value = {
        "kind": "cpp_phase_fold_reference",
        "version": 1,
        "profile": "exact-phase-fold-word-v1",
        "sdk": build_info(),
        "manifest": archive.manifest,
        "limits": archive.limits,
        "history": history,
    }
    validate_phase_fold_reference(value)
    return value


def phase_fold_record(archive):
    """Retain the complete RFCXW word and any selected native publication prefix."""
    from rexgraph.graph import RexGraph

    reference = phase_fold_reference(archive)
    record = RexGraph.from_hypergraph([0, 1], [0])
    record.attach_metadata(
        1,
        0,
        "phase_fold_archive",
        {
            "kind": "cpp_phase_fold_archive",
            "version": 1,
            "reference": reference,
            "archive": np.frombuffer(archive.archive, np.uint8).copy(),
        },
    )
    if archive.history_owner is not None:
        record.attach_metadata(
            1,
            0,
            "phase_fold_history",
            {
                "kind": "cpp_phase_fold_history",
                "version": 1,
                "reference": reference["history"],
                "archive": np.frombuffer(archive.history_owner.archive, np.uint8).copy(),
            },
        )
    return record


def _bytes(attachment, bound, digest):
    raw = attachment["archive"]
    if (
        not isinstance(raw, np.ndarray)
        or raw.dtype != np.uint8
        or raw.ndim != 1
        or not 32 <= raw.size <= bound
    ):
        raise ValueError("phase/fold archive tensor differs or exceeds byte admission")
    if sha256(raw.tobytes()).hexdigest() != digest:
        raise ValueError("phase/fold archive bytes differ")
    return raw


def _check_record(record, index=0, *, max_archive_bytes=256 << 20, history_max_bytes=256 << 20):
    index = _key(index)
    attachment = record.get_metadata(1, index, "phase_fold_archive")
    _closed(attachment, {"kind", "version", "reference", "archive"})
    if (
        attachment["kind"] != "cpp_phase_fold_archive"
        or type(attachment["version"]) is not int
        or attachment["version"] != 1
    ):
        raise ValueError("unsupported phase/fold archive attachment")
    ref = validate_phase_fold_reference(attachment["reference"])
    raw = _bytes(
        attachment,
        min(_key(max_archive_bytes), ref["limits"]["wire.max_file_bytes"]),
        ref["manifest"]["archive_digest"],
    )
    checked = None
    if ref["history"] is not None:
        h = record.get_metadata(1, index, "phase_fold_history")
        _closed(h, {"kind", "version", "reference", "archive"})
        if (
            h["kind"] != "cpp_phase_fold_history"
            or type(h["version"]) is not int
            or h["version"] != 1
            or h["reference"] != ref["history"]
        ):
            raise ValueError("phase/fold history attachment differs")
        source = _history(ref["history"], ref["manifest"])
        bound = min(_key(history_max_bytes), ref["history"]["max_bytes"])
        checked = (_bytes(h, bound, source.native.archive_digest), source, bound)
    return ref, raw, checked


def read_phase_fold_record(
    record, index=0, *, limits=None, history_max_bytes=256 << 20, _archive_cache=None
):
    """Restore canonical native owners under intersected saved and reader limits."""
    from rexgraph_cpp import NativePhaseFoldArchive, NativeSource

    from rexgraph.io.algebraic_record import _AlgebraicRestoreCache

    reader = NativePhaseFoldArchive.default_limits()
    if limits is not None:
        if not isinstance(limits, Mapping) or set(limits) - LIMIT_NAMES:
            raise ValueError("unknown phase/fold reader limits")
        reader.update({k: _key(v) for k, v in limits.items()})
    ref, raw, checked = _check_record(
        record,
        index,
        max_archive_bytes=reader["wire.max_file_bytes"],
        history_max_bytes=history_max_bytes,
    )
    effective = {k: min(v, ref["limits"][k]) for k, v in reader.items()}
    if _archive_cache is not None and not isinstance(_archive_cache, _AlgebraicRestoreCache):
        raise TypeError("phase/fold cache requires its bounded archive restore implementation")
    key = (
        "exact-phase-fold-word-v1",
        _key(index),
        sha256(pack_value(ref)).hexdigest(),
        tuple(sorted(effective.items())),
        None if checked is None else (checked[1].coefficient_digest, checked[2], id(checked[0])),
    )
    owner = None if _archive_cache is None else _archive_cache.get(key, record, raw)
    if owner is None:
        if checked is None:
            owner = NativePhaseFoldArchive(raw.tobytes(), limits=effective)
        else:
            tensor, source, bound = checked
            native = NativeSource(tensor.tobytes(), max_bytes=bound)
            source.bind(native)
            owner = NativePhaseFoldArchive.from_history(
                native, ref["history"]["channel"], sequence=source.native.sequence, limits=effective
            )
        expected = deepcopy(ref)
        expected["limits"] = effective
        if checked is not None:
            expected["history"]["max_bytes"] = checked[2]
        if phase_fold_reference(owner) != expected:
            raise ValueError("phase/fold native declaration differs from retained archive")
        if _archive_cache is not None:
            owner = _archive_cache.put(
                key, record, raw, owner, int(raw.size + (0 if checked is None else checked[0].size))
            )
    return owner


def _observation(selected, max_coordinates):
    value = {
        "kind": "cpp_phase_fold_observation",
        "version": 1,
        "profile": "exact-phase-fold-endpoint-v1",
        "source": phase_fold_reference(selected),
        "endpoint": selected.observe(max_coordinates=max_coordinates),
    }
    value["seal"] = sha256(pack_value(value)).hexdigest()
    validate_phase_fold_observation(value, max_coordinates=max_coordinates)
    return value


def phase_fold_observation(archive, count, *, max_coordinates=65536):
    """Observe a complete logical prefix with no pending edits or later operations."""
    return _observation(archive.observed_prefix(count), max_coordinates)


def validate_phase_fold_observation(value, *, max_coordinates=65536):
    """Validate exact endpoint coordinates, retained operation order and the observation seal."""
    _closed(value, {"kind", "version", "profile", "source", "endpoint", "seal"})
    if (
        value["kind"] != "cpp_phase_fold_observation"
        or type(value["version"]) is not int
        or value["version"] != 1
        or value["profile"] != "exact-phase-fold-endpoint-v1"
    ):
        raise ValueError("unsupported phase/fold observation")
    ref = validate_phase_fold_reference(value["source"])
    m = ref["manifest"]
    if m["proposal_count"] or ref["history"] is not None:
        raise ValueError("logical observation retains prospective or publication channels")
    if m["observation_coordinate_count"] > _key(max_coordinates):
        raise ValueError("phase/fold observation exceeds its complete coordinate admission")
    endpoint = value["endpoint"]
    _closed(
        endpoint,
        (MANIFEST_NAMES - {"codec", "archive_digest", "source_digest"})
        | {"endpoint", "operations", "positions"},
    )
    if any(
        endpoint[k] != v
        for k, v in m.items()
        if k not in {"codec", "archive_digest", "source_digest"}
    ):
        raise ValueError("phase/fold observation declaration differs")
    numbers = endpoint["endpoint"]
    integer_bits = ref["limits"]["wire.max_integer_bytes"] * 8
    _coordinates(numbers, m["endpoint_coordinate_count"], integer_bits=integer_bits)
    operations = endpoint["operations"]
    if not isinstance(operations, (tuple, list)) or len(operations) != m["operation_count"]:
        raise ValueError("phase/fold observation loses retained operations")
    coordinate_count = len(numbers) + len(m["minimal_polynomial"]) + len(m["involution"])
    for op in operations:
        walk = m["word_kind"].endswith("walk")
        names = (
            {"edge", "forward"}
            if walk
            else {"identity", "active", "phase"}
            | ({"crease", "moving_panels"} if m["word_kind"] == "rational-fold" else set())
        )
        _closed(op, names)
        if walk:
            if _key(op["edge"]) >= m["source_sizes"][1] or type(op["forward"]) is not bool:
                raise ValueError("invalid retained phase walk step")
        else:
            _key(op["identity"])
            if type(op["active"]) is not bool:
                raise ValueError("invalid retained phase/fold operation")
            _coordinates(
                op["phase"],
                m["field_degree"] if m["word_kind"] == "algebraic-phase" else 2,
                integer_bits=min(integer_bits, ref["limits"]["history.max_integer_bits"]),
            )
            coordinate_count += len(op["phase"])
            if m["word_kind"] == "rational-fold":
                if _key(op["crease"]) >= ref["limits"]["panels.max_creases"]:
                    raise ValueError("fold crease exceeds declared admission")
                if (
                    not isinstance(op["moving_panels"], (tuple, list))
                    or not op["moving_panels"]
                    or any(_key(n) >= m["source_sizes"][2] for n in op["moving_panels"])
                ):
                    raise ValueError("invalid fold moving group")
                if len(set(op["moving_panels"])) != len(op["moving_panels"]):
                    raise ValueError("fold moving group repeats a panel")
                coordinate_count += len(op["moving_panels"])
    positions = endpoint["positions"]
    if not isinstance(positions, (tuple, list)):
        raise ValueError("phase walk lift positions require a retained sequence")
    if m["word_kind"].endswith("walk"):
        if len(positions) != m["operation_count"] + 1 or any(type(n) is not int for n in positions):
            raise ValueError("incomplete exact phase walk lift")
        if any(
            max(1, abs(n).bit_length()) > ref["limits"]["cover.max_integer_bits"] for n in positions
        ):
            raise ValueError("phase walk lift exceeds its integer admission")
        coordinate_count += len(positions)
    elif positions:
        raise ValueError("nonwalk observation has lift positions")
    if coordinate_count != m["observation_coordinate_count"] or coordinate_count > _key(
        max_coordinates
    ):
        raise ValueError("phase/fold observation exceeds its complete coordinate admission")
    _sha(value["seal"])
    if (
        sha256(pack_value({k: v for k, v in value.items() if k != "seal"})).hexdigest()
        != value["seal"]
    ):
        raise ValueError("phase/fold observation seal differs")
    return value


def phase_fold_observation_record(archive, count, *, max_coordinates=65536):
    """Save a prefix's exact observation beside only its complete observed word."""
    selected = archive.observed_prefix(count)
    observation = _observation(selected, max_coordinates)
    record = phase_fold_record(selected)
    record.attach_metadata(1, 0, "phase_fold_observation", observation)
    return record


def read_phase_fold_observation_record(
    record,
    index=0,
    *,
    max_archive_bytes=256 << 20,
    history_max_bytes=256 << 20,
    max_coordinates=65536,
):
    """Read a checked retained observation without importing the native engine."""
    ref, _, _ = _check_record(
        record, index, max_archive_bytes=max_archive_bytes, history_max_bytes=history_max_bytes
    )
    value = record.get_metadata(1, _key(index), "phase_fold_observation")
    validate_phase_fold_observation(value, max_coordinates=max_coordinates)
    if value["source"] != ref:
        raise ValueError("phase/fold observation differs from its retained word")
    return deepcopy(value)


def phase_fold_history_record(source, channel, *, sequence, limits=None):
    """Retain one RFCXW channel with its complete selected native publication prefix."""
    from rexgraph_cpp import NativePhaseFoldArchive

    return phase_fold_record(
        NativePhaseFoldArchive.from_history(source, channel, sequence=sequence, limits=limits)
    )


def phase_fold_history_at(record, sequence, *, limits=None, history_max_bytes=256 << 20):
    """Select an earlier native publication independently of its logical operation count."""
    from rexgraph_cpp import NativePhaseFoldArchive

    archive = read_phase_fold_record(record, limits=limits, history_max_bytes=history_max_bytes)
    if archive.history_owner is None:
        raise ValueError("phase/fold record has no retained native publication")
    return phase_fold_record(
        NativePhaseFoldArchive.from_history(
            archive.history_owner, archive.history_channel, sequence=sequence, limits=archive.limits
        )
    )
