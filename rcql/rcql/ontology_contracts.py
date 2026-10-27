"""Prospective ontology types; planning does not open providers or solve axioms."""
from functools import lru_cache
import sys

from .types import RCType, ValueKind, Exactness

ARGUMENTS = {
    "ONTOLOGY_READ": (("grade", "index"), (1, 0)),
    "ONTOLOGY_IMPORT": (("archive",), ()),
    "ONTOLOGY_EXPORT": (("ontology",), ()),
    "ONTOLOGY_RECORD": (("ontology",), ()),
    "ONTOLOGY_INFO": (("ontology",), ()),
    "ONTOLOGY_DECLARATION": (("ontology",), ()),
    "ONTOLOGY_AT": (("ontology", "cutoff"), ()),
    "ONTOLOGY_REVISE": (("ontology", "graph", "changes", "sources"), (None,)),
    "ONTOLOGY_VERIFY": (("ontology", "cutoff", "sources", "parent"), (None, None)),
    "ONTOLOGY_NATIVE_APPEND": (("ontology", "component", "frame", "cutoff", "sources"), (None,)),
    "ONTOLOGY_NATIVE_ADD": (("ontology", "component", "specification", "cutoff", "sources"), (None,)),
}


@lru_cache(maxsize=None)
def _graph_pattern(base):
    class GraphPattern(base):
        def accepts(self, value, *, source=None):
            if isinstance(value, RCType):
                return super().accepts(value, source=source)
            module = sys.modules.get("rexgraph.graph")
            graph_type = getattr(module, "RexGraph", None)
            return graph_type is not None and isinstance(value, graph_type)
    return GraphPattern


def install(register, OperatorSignature, TypePattern):
    v = ValueKind
    table = {
        "ONTOLOGY_READ": ((int, int), v.ONTOLOGY),
        "ONTOLOGY_IMPORT": ((bytes,), v.ONTOLOGY),
        "ONTOLOGY_EXPORT": ((v.ONTOLOGY,), v.BYTES),
        "ONTOLOGY_RECORD": ((v.ONTOLOGY,), v.REX),
        "ONTOLOGY_INFO": ((v.ONTOLOGY,), v.RECORD),
        "ONTOLOGY_DECLARATION": ((v.ONTOLOGY,), v.RECORD),
        "ONTOLOGY_AT": ((v.ONTOLOGY, None), v.ONTOLOGY),
        "ONTOLOGY_REVISE": ((v.ONTOLOGY, v.REX, dict, None), v.ONTOLOGY),
        "ONTOLOGY_VERIFY": ((v.ONTOLOGY, None, None, None), v.RECORD),
        "ONTOLOGY_NATIVE_APPEND": ((v.ONTOLOGY, str, bytes, None, None), v.ONTOLOGY),
        "ONTOLOGY_NATIVE_ADD": ((v.ONTOLOGY, str, dict, None, None), v.ONTOLOGY),
    }
    for name, (labels, defaults) in ARGUMENTS.items():
        kinds, output = table[name]
        patterns = tuple((_graph_pattern(TypePattern) if kind is v.REX else TypePattern)(label, kind=kind,
                                    optional=i >= len(labels)-len(defaults))
                         if isinstance(kind, ValueKind) else
                         TypePattern(label, literal=kind, optional=i >= len(labels)-len(defaults))
                         for i, (label, kind) in enumerate(zip(labels, kinds, strict=True)))
        register(OperatorSignature(name, v.REX, patterns,
            RCType(output.value, kind=output, exactness=Exactness.STRUCTURAL),
            "rcql.ontology_operators." + name.lower(),
            requires=frozenset({"read", "train"} if name in
                              {"ONTOLOGY_REVISE", "ONTOLOGY_NATIVE_APPEND", "ONTOLOGY_NATIVE_ADD"}
                              else {"read"}),
            memoizable=False))


def literal_type(binding, value):
    module = sys.modules.get("rexgraph.ontology")
    ontology_type = getattr(module, "OntologyState", None)
    if ontology_type is not None and isinstance(value, ontology_type):
        return RCType("OntologyState", kind=ValueKind.ONTOLOGY,
            exactness=Exactness.STRUCTURAL, declaration_digest=value.digest)
    return None
