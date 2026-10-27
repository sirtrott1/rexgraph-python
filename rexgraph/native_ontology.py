"""Optional native source-history validation for explicitly enclosed components."""
from __future__ import annotations

from .ontology import OntologyProvider, register_ontology_provider
from .source_binding import CppSourceIdentity

PROVIDER = {"name": "rexgraph.native-ontology-source", "version": 1}
_ENCLOSURE = {"kind": "enclosure", "scope": "independent-native-component"}


def native_ontology_component(key, source):
    """Retain an actual owning native series without asserting a graph morphism."""
    from rexgraph_cpp import NativeSource
    if type(key) is not str or not key or len(key.encode("utf-8")) > 4096:
        raise ValueError("native ontology component requires a bounded identity")
    if not isinstance(source, NativeSource):
        raise TypeError("native ontology component requires an owning NativeSource")
    return {"key": key, "archive": source.archive, "identity": source.identity,
            "accession": dict(_ENCLOSURE)}


def _components(ontology):
    from rexgraph_cpp import NativeSource
    components = ontology.declaration["signature"].get("native_sources")
    if type(components) is not tuple or not 0 < len(components) <= 64:
        raise ValueError("native ontology profile requires bounded ordered native_sources")
    owners = {}
    for component in components:
        if (type(component) is not dict or set(component) != {"key", "archive", "identity", "accession"}
                or component["accession"] != _ENCLOSURE):
            raise ValueError("native ontology component requires its explicit enclosure scope")
        key, raw = component["key"], component["archive"]
        if (type(key) is not str or not key or "\x00" in key
                or len(key.encode("utf-8")) > 4096 or key in owners):
            raise ValueError("native ontology component identities must be nonempty and distinct")
        if type(raw) is not bytes or not 0 < len(raw) <= 64 << 20:
            raise ValueError("native ontology source exceeds byte admission")
        expected = CppSourceIdentity.from_record(component["identity"])
        owner = NativeSource(raw, max_bytes=64 << 20, max_dimension=1_000_000,
                             max_records=1_000_000, max_logical_size=16_000_000)
        if owner.archive != raw or CppSourceIdentity.from_record(owner.identity) != expected:
            raise ValueError("native ontology source differs from its actual publication")
        owners[key] = owner
    return owners


def _verify(ontology, *, sources, parent):
    """Check the declared scope; neither source access nor arbitrary axioms follow."""
    owners = _components(ontology)
    parents = {} if parent is None else _components(parent)
    if parents and owners.keys() != parents.keys():
        raise ValueError("native prefix profile requires the same explicitly enclosed components")
    for key, old in parents.items():
        new = owners[key]
        a, b = old.identity, new.identity
        if (a["namespace"] != b["namespace"] or a["first_sequence"] != b["first_sequence"]
                or a["sequence"] > b["sequence"]
                or new.at(a["sequence"]).archive != old.at(a["sequence"]).archive):
            raise ValueError("native ontology child does not retain its actual parent prefix")
    return {"profile": "rexgraph.native-ontology-source.v1", "ontology": ontology.digest,
        "format_integrity": "supported", "native_source_integrity": "supported",
        "components": tuple({"key": key, "identity": owner.identity} for key, owner in owners.items()),
        "source_accession": {"enclosure": "supported", "graph_to_native_morphism": "not_evaluated"},
        "transport_scope": "native-series-replay-and-exact-parent-prefix",
        "axiom_profile_result": "not_evaluated", "current_availability": "not_evaluated"}


def register_native_ontology_provider():
    """Install this bounded native profile explicitly in the receiving process."""
    return register_ontology_provider(OntologyProvider(PROVIDER["name"], PROVIDER["version"], _verify))


def _evolve(ontology, component, operation, *, parent_source, known_at):
    from hashlib import sha256
    from .ontology import OntologyState
    from .tensor_field import FieldSource
    from rexgraph_cpp._source_evolution import NativeSourceEvolution

    if not isinstance(ontology, OntologyState):
        raise TypeError("native revision requires an actual parent ontology")
    if not isinstance(parent_source, FieldSource) or parent_source.source is None:
        raise TypeError("native revision requires the actual bound parent source")
    parent_source.check()
    if parent_source.state_digest != ontology.source_digest:
        raise ValueError("native revision parent differs from the selected source")
    if parent_source.record_id is None or parent_source.version is None or parent_source.version < 1:
        raise ValueError("native revision requires a positive parent record version")
    declaration = ontology.declaration
    if declaration["provider"] != PROVIDER:
        raise ValueError("native revision requires its declared source-history profile")
    owners = _components(ontology)
    if component not in owners:
        raise ValueError("native revision requires an existing component identity")
    base = NativeSourceEvolution(owners[component].archive)
    revised, kind = operation(base)
    selected = native_ontology_component(component, revised.source())
    signature = declaration["signature"]
    signature["native_sources"] = tuple(selected if row["key"] == component else row
                                         for row in signature["native_sources"])
    dependencies = list(ontology.dependencies)
    if not any(ref.matches(parent_source) for ref in dependencies):
        dependencies.append(parent_source.detached())
    accession = {"profile": "native-series-interval-v1", "component": component,
        "from_sequence": base.identity[2], "to_sequence": revised.identity[2],
        "parent_archive": sha256(base.archive).hexdigest(),
        "child_archive": sha256(revised.archive).hexdigest(),
        "scope": "independent-native-component"}
    return ontology.revise(ontology.record(), known_at=known_at, kind=kind,
        maps=(accession,), signature=signature, dependencies=tuple(dependencies))


def append_native_ontology(ontology, component, frame, *, parent_source, known_at):
    """Append an actual native frame and retain its complete source history."""
    return _evolve(ontology, component, lambda owner: (owner.append(frame), "ontology-revision"),
                   parent_source=parent_source, known_at=known_at)


def add_native_ontology_declarations(ontology, component, specification, *, parent_source, known_at):
    """Add explicitly filled nonincidence aspects or declared exact types."""
    return _evolve(ontology, component,
        lambda owner: (owner.add_declarations(specification), "declaration-recode"),
        parent_source=parent_source, known_at=known_at)


__all__ = ["PROVIDER", "native_ontology_component", "register_native_ontology_provider",
           "append_native_ontology", "add_native_ontology_declarations"]
