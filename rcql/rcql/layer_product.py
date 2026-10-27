"""Retained finite layer diagrams, exact compatibility and revisable operators."""
from __future__ import annotations

import base64
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
import hashlib

from rexgraph.coordinate_map import CoordinateMap
from rexgraph.exact_value import exact_fraction
from rexgraph.tensor_field import FieldSource, TensorField
from rexgraph.type_accession import CoordinateSpace
from rexgraph.source_binding import source_from_record
from rexgraph.value_codec import pack_value, unpack_value

from .native_temporal import TemporalModel
from .program import Program

_LIMIT = 64*1024*1024
_MAGIC = b"RGLY\x01"


def _pack(kind, value):
    body = pack_value({"version": 1, "kind": kind, "value": value})
    raw = _MAGIC+hashlib.sha256(body).digest()+body
    if len(raw) > _LIMIT:
        raise ValueError("layer declaration exceeds its byte limit")
    return raw


def _unpack(raw, kind):
    if type(raw) is not bytes or len(raw) > _LIMIT or not raw.startswith(_MAGIC) or len(raw) < 37:
        raise ValueError("invalid layer declaration framing")
    if hashlib.sha256(raw[37:]).digest() != raw[5:37]:
        raise ValueError("layer declaration digest mismatch")
    value = unpack_value(raw[37:])
    if (not isinstance(value, dict) or set(value) != {"version", "kind", "value"}
            or type(value["version"]) is not int or value["version"] != 1 or value["kind"] != kind):
        raise ValueError("unknown layer declaration schema")
    if _pack(kind, value["value"]) != raw:
        raise ValueError("noncanonical layer declaration")
    return value["value"]


def _name(value):
    if not isinstance(value, str) or not value or len(value.encode()) > 65536:
        raise ValueError("layer identities and roles require bounded nonempty text")
    return value


def _space(value):
    if not isinstance(value, CoordinateSpace):
        raise TypeError("layer carrier requires a named CoordinateSpace")
    if len(value.keys) > 65536:
        raise ValueError("layer carrier exceeds its coordinate limit")
    return value.name, value.keys


def _map(value):
    if not isinstance(value, CoordinateMap):
        raise TypeError("layer restrictions require explicit CoordinateMaps")
    if len(value.entries) > 1000000:
        raise ValueError("layer restriction entry limit")
    return _space(value.domain), _space(value.codomain), value.entries


def _read_map(value):
    a, b, entries = value
    return CoordinateMap(CoordinateSpace(*a), CoordinateSpace(*b), entries)


def _record(value, key):
    from rexgraph import RexGraph
    result = RexGraph.from_cells([1, []])
    result.attach_metadata(0, 0, key, base64.b64encode(value.data).decode("ascii"))
    if getattr(value, "dependencies", ()):
        result._agent_meta = {"rcql_evidence": [v.as_record() for v in value.dependencies]}
    return result


def _read_record(cls, record, key):
    from rexgraph.object_identity import object_digest
    raw = record.get_metadata(0, 0, key)
    if not isinstance(raw, str):
        raise ValueError("record has no requested layer declaration")
    value = cls(base64.b64decode(raw, validate=True))
    if object_digest(_record(value, key)) != object_digest(record):
        raise ValueError("layer record carrier differs from its declaration")
    return value


class _Value:
    @property
    def dependencies(self):
        return ()

    @property
    def coefficient_digest(self):
        return hashlib.sha256(self.data).hexdigest()

    def to_bytes(self):
        return self.data

    @classmethod
    def from_bytes(cls, raw):
        return cls(raw)

    def to_record(self):
        return _record(self, "rcql_"+self._kind)

    @classmethod
    def from_record(cls, record):
        return _read_record(cls, record, "rcql_"+cls._kind)


@dataclass(frozen=True)
class Layer(_Value):
    """One identified carrier or native function/family with a participation role."""
    data: bytes
    _kind = "layer"

    def __post_init__(self):
        d = self.declaration
        if set(d) != {"key", "role", "profile", "space", "program", "coefficients", "inputs", "native", "object", "support", "scale", "shift", "native_program"}:
            raise ValueError("invalid layer fields")
        _name(d["key"]); _name(d["role"])
        if d["profile"] == "affine":
            space = CoordinateSpace(*d["space"]); _space(space)
            if d["program"] is not None:
                program = Program.from_bytes(d["program"])
                ports = {v.name for v in program.inputs}
                if any(p not in ports or c not in space.keys for p, c in d["coefficients"]):
                    raise ValueError("program coefficient port/coordinate mismatch")
            elif d["coefficients"] or d["inputs"]:
                raise ValueError("only program layers have executable ports")
            if d["native"] is not None or d["object"] or d["support"] or d["native_program"]:
                raise ValueError("affine layers cannot contain native function handles")
        elif d["profile"] in {"functions", "families"}:
            TemporalModel(d["native"])
            if any(type(d[k]) is not int or not 0 <= d[k] < 2**64 for k in ("object", "support", "native_program")):
                raise ValueError("native layer identity outside its source")
            if not d["object"] and not d["native_program"]:
                raise ValueError("native layer requires an object or program")
            if d["profile"] == "functions" and not d["support"]:
                raise ValueError("function layer requires its declared cover support")
            if d["space"] is not None or d["program"] is not None or d["coefficients"] or d["inputs"]:
                raise ValueError("native layers cannot reinterpret affine program ports")
        elif d["profile"] == "c_workflows":
            from .c_workflow import CWorkflow
            CWorkflow.from_bytes(d["native"]).value(d["object"])
            if d["space"] is not None or d["program"] is not None or d["coefficients"] or d["inputs"] or d["support"] or d["native_program"] or d["scale"] != 1 or d["shift"]:
                raise ValueError("C layers retain their original native declarations")
        else:
            raise ValueError("unsupported layer profile")
        exact_fraction(d["scale"]); exact_fraction(d["shift"])

    @property
    def declaration(self):
        return _unpack(self.data, self._kind)

    @property
    def dependencies(self):
        raw = self.declaration["native"]
        if self.profile == "c_workflows":
            from .c_workflow import CWorkflow
            return CWorkflow.from_bytes(raw).dependencies
        if raw is not None:
            return TemporalModel(raw).dependencies
        program = self.declaration["program"]
        if program is None:
            return ()
        from .source_context import field_references
        return field_references(Program.from_bytes(program))

    @property
    def key(self):
        return self.declaration["key"]

    @property
    def profile(self):
        return self.declaration["profile"]

    @classmethod
    def affine(cls, key, space, *, role="field", program=None, coefficients=(), inputs=()):
        return cls(_pack(cls._kind, {"key": key, "role": role, "profile": "affine", "space": _space(space),
            "program": None if program is None else program.to_bytes(), "coefficients": tuple(tuple(v) for v in coefficients),
            "inputs": tuple(tuple(v) for v in inputs), "native": None, "object": 0, "support": 0,
            "scale": Fraction(1), "shift": Fraction(0), "native_program": 0}))

    @classmethod
    def c_value(cls, key, value, *, role="participant"):
        from .c_workflow import CValue
        if type(value) is not CValue:
            raise TypeError("a typed native C value is required")
        result=cls(_pack(cls._kind, {"key":key,"role":role,"profile":"c_workflows",
            "space":None,"program":None,"coefficients":(),"inputs":(),
            "native":value.workflow.data,"object":value.index,"support":0,
            "scale":Fraction(1),"shift":Fraction(0),"native_program":0}))
        object.__setattr__(result,"_c_owner",value.workflow)
        return result

    @classmethod
    def function(cls, key, model, support, *, function=None, program=0, scale=1, shift=0, role="function"):
        if not isinstance(model, TemporalModel):
            raise TypeError("function layers require the actual retained native source")
        return cls(_pack(cls._kind, {"key": key, "role": role, "profile": "functions", "space": None,
            "program": None, "coefficients": (), "inputs": (), "native": model.data,
            "object": model.selected if function is None and not program else 0 if function is None else function,
            "support": support, "scale": exact_fraction(scale), "shift": exact_fraction(shift), "native_program": program}))

    @classmethod
    def family(cls, key, model, family=None, *, role="context"):
        if not isinstance(model, TemporalModel):
            raise TypeError("finite layers require the actual retained native source")
        return cls(_pack(cls._kind, {"key": key, "role": role, "profile": "families", "space": None,
            "program": None, "coefficients": (), "inputs": (), "native": model.data,
            "object": model.selected if family is None else family, "support": 0,
            "scale": Fraction(1), "shift": Fraction(0), "native_program": 0}))


@dataclass(frozen=True)
class LayerInterface(_Value):
    """An identified equality interface with an actual map from every participant."""
    data: bytes
    _kind = "layer_interface"

    def __post_init__(self):
        d = self.declaration
        if set(d) != {"key", "maps"}:
            raise ValueError("invalid interface fields")
        _name(d["key"])
        maps = tuple(d["maps"])
        if not 2 <= len(maps) <= 4096 or len({n for n, _ in maps}) != len(maps):
            raise ValueError("affine interface requires distinct participating layers")
        values = [_read_map(m) for _, m in maps]
        if any(m.codomain != values[0].codomain for m in values):
            raise ValueError("interface maps require the same declared target carrier")
        for key, _ in maps:
            _name(key)

    @property
    def declaration(self):
        return _unpack(self.data, self._kind)

    @classmethod
    def equality(cls, key, maps):
        return cls(_pack(cls._kind, {"key": key, "maps": tuple((n, _map(m)) for n, m in maps)}))


@dataclass(frozen=True)
class LayerProduct(_Value):
    """The retained diagram, not a Cartesian product of projected value sets."""
    data: bytes
    _kind = "layer_product"

    def __post_init__(self):
        d = self.declaration
        if set(d) != {"name", "layers", "interfaces", "links", "outputs", "transitions", "dependencies", "parent"}:
            raise ValueError("invalid layer-product fields")
        _name(d["name"])
        layers = self.layers
        if not 1 <= len(layers) <= 4096 or len({v.key for v in layers}) != len(layers):
            raise ValueError("layer product requires distinct declared layer identities")
        if len({v.profile for v in layers}) != 1:
            raise NotImplementedError("cross-profile layer maps require an explicit admitted adapter")
        for ref in self.dependencies:
            ref.check()
        declared_dependencies = {v.coefficient_digest for v in self.dependencies}
        if any(ref.coefficient_digest not in declared_dependencies for layer in layers for ref in layer.dependencies):
            raise ValueError("layer product discarded a layer source dependency")
        parent, depth = d["parent"], 0
        while parent is not None:
            depth += 1
            if depth > 128:
                raise ValueError("layer revision depth limit")
            old = _unpack(parent, self._kind)
            if set(old) != set(d) or any(ref not in d["dependencies"] for ref in old["dependencies"]):
                raise ValueError("layer revision discarded a retained source dependency")
            parent = old["parent"]
        if self.profile == "affine":
            spaces = {v.key: CoordinateSpace(*v.declaration["space"]) for v in layers}
            if sum(len(v.keys) for v in spaces.values()) > 65536:
                raise ValueError("layer-product coordinate limit")
            interfaces = [LayerInterface(raw) for raw in d["interfaces"]]
            if len(interfaces) > 4096 or len({v.declaration["key"] for v in interfaces}) != len(interfaces):
                raise ValueError("interface identity/size limit")
            for interface in interfaces:
                for key, declaration in interface.declaration["maps"]:
                    if key not in spaces or _read_map(declaration).domain != spaces[key]:
                        raise ValueError("restriction domain differs from its actual layer carrier")
            if d["transitions"]:
                raise ValueError("affine products use declared interface maps")
            self.assembly()
        elif self.profile == "c_workflows":
            from .c_workflow import validate_product
            if d["outputs"]:
                raise ValueError("C products return the complete declared native interface")
            validate_product(self)
        else:
            if d["interfaces"] or d["links"] or d["outputs"]:
                raise ValueError("native products use their native charts/transitions")
            natives = [TemporalModel(v.declaration["native"]) for v in layers]
            first = natives[0].declaration
            if any((n.declaration["source"], n.declaration["endpoint"], n.declaration["limits"], n.declaration["build"]) !=
                   (first["source"], first["endpoint"], first["limits"], first["build"]) for n in natives[1:]):
                raise ValueError("native layer objects require the same actual source snapshot")
            if self.profile == "functions" and d["transitions"]:
                raise ValueError("function products use their declared cover charts")

    @property
    def declaration(self):
        return _unpack(self.data, self._kind)

    @property
    def layers(self):
        result=tuple(Layer(v) for v in self.declaration["layers"])
        owner=getattr(self,"_c_owner",None)
        if owner is not None:
            for layer in result:object.__setattr__(layer,"_c_owner",owner)
        return result

    @property
    def profile(self):
        return self.layers[0].profile

    @property
    def dependencies(self):
        return tuple(source_from_record(v, allow_cpp=True) for v in self.declaration["dependencies"])

    @classmethod
    def create(cls, layers, interfaces=(), *, name="layers", links=(), outputs=(), transitions=(), dependencies=()):
        layers, interfaces = tuple(layers), tuple(interfaces)
        from .c_workflow import CInterface, shared_owner
        c_profile=bool(layers) and layers[0].profile == "c_workflows"
        owner=shared_owner(layers,interfaces) if c_profile else None
        interface_type=CInterface if c_profile else LayerInterface
        if any(not isinstance(v, Layer) for v in layers) or any(not isinstance(v, interface_type) for v in interfaces):
            raise TypeError("typed layers and interfaces required")
        refs = {}
        dependencies = (*dependencies, *(ref for layer in layers for ref in layer.dependencies))
        for ref in dependencies:
            if not isinstance(ref, FieldSource):
                raise TypeError("layer dependencies require exact source references")
            refs[ref.coefficient_digest] = ref.as_record()
        result=cls(_pack(cls._kind, {"name": name, "layers": tuple(v.data for v in layers),
            "interfaces": tuple(v.data for v in interfaces), "links": tuple(tuple(v) for v in links),
            "outputs": tuple(tuple(v) for v in outputs), "transitions": tuple(dict(v) for v in transitions),
            "dependencies": tuple(refs.values()), "parent": None}))
        if owner is not None:object.__setattr__(result,"_c_owner",owner)
        return result

    def modify(self, replacements=(), interfaces=None):
        if interfaces is not None:interfaces=tuple(interfaces)
        replacements = tuple(replacements)
        if any(not isinstance(v, Layer) for v in replacements) or len({v.key for v in replacements}) != len(replacements):
            raise ValueError("layer revisions require distinct typed replacements")
        by_key = {v.key: v for v in self.layers}
        if set(v.key for v in replacements)-set(by_key):
            raise ValueError("layer revision names an absent layer")
        by_key.update((v.key, v) for v in replacements)
        d = self.declaration
        refs = {ref.coefficient_digest: ref.as_record() for ref in self.dependencies}
        for layer in by_key.values():
            for ref in layer.dependencies:
                refs[ref.coefficient_digest] = ref.as_record()
        revised = {**d, "layers": tuple(by_key[v.key].data for v in self.layers), "parent": self.data,
                   "dependencies": tuple(refs.values())}
        from .c_workflow import CInterface, shared_owner, validate_product
        interface_type=CInterface if self.profile == "c_workflows" else LayerInterface
        if self.profile == "c_workflows":
            owner=validate_product(self)
            for layer in by_key.values():
                if layer.key not in {v.key for v in replacements}:object.__setattr__(layer,"_c_owner",owner)
            if interfaces is None:
                interfaces=tuple(CInterface(v) for v in d["interfaces"])
                for interface in interfaces:object.__setattr__(interface,"_c_owner",owner)
            owner=shared_owner(tuple(by_key.values()),tuple(interfaces))
        if interfaces is not None:
            interfaces = tuple(interfaces)
            revised["interfaces"] = tuple(v.data for v in interfaces if isinstance(v, interface_type))
            if len(revised["interfaces"]) != len(interfaces):
                raise TypeError("revised interfaces require typed declarations")
        result=LayerProduct(_pack(self._kind, revised))
        if self.profile == "c_workflows":object.__setattr__(result,"_c_owner",owner)
        return result

    def system(self, sources=None):
        if self.profile != "affine":
            raise TypeError("only the affine profile has a section system")
        from rexgraph import RexGraph
        from rexgraph.section_calculus import SectionRecipe, SectionSystem
        layers = self.layers
        names = tuple(v.key for v in layers)
        interfaces = tuple(LayerInterface(raw).declaration for raw in self.declaration["interfaces"])
        indices = {key: i for i, key in enumerate(names)}
        supports = [[indices[key] for key, _ in interface["maps"]] for interface in interfaces]
        # These cells are the identified mathematical layers of this diagram.
        # Their original field carriers and participation roles stay in metadata.
        source = RexGraph.from_cells([len(layers), supports])
        source.attach_metadata(0, 0, "rcql_layer_diagram", base64.b64encode(self.data).decode("ascii"))
        spaces = tuple(CoordinateSpace(*v.declaration["space"]) for v in layers)
        targets, restrictions, incidence = [], [], [[] for _ in layers]
        for m, interface in enumerate(interfaces):
            maps = [(indices[key], _read_map(raw)) for key, raw in interface["maps"]]
            targets.append(maps[0][1].codomain)
            for c, action in maps:
                incidence[c].append(m); restrictions.append((c, m, action))
        recipe = SectionRecipe(self.declaration["name"], 0, names,
            tuple(v["key"] for v in interfaces), spaces, tuple(targets), tuple(tuple(v) for v in incidence),
            tuple(restrictions), FieldSource(source), _bound_dependencies(self, sources))
        return SectionSystem(recipe)

    def assembly(self):
        from .program_family import ProgramAssembly
        if self.profile != "affine":
            return None
        layers = self.layers
        fragments = [(v.key, Program.from_bytes(v.declaration["program"])) for v in layers if v.declaration["program"] is not None]
        if not fragments:
            if self.declaration["links"] or self.declaration["outputs"]:
                raise ValueError("only executable layers have program links/outputs")
            return None
        result = ProgramAssembly.create(self.declaration["name"], fragments,
            tuple((v.key, port, coordinate) for v in layers for port, coordinate in v.declaration["coefficients"]),
            tuple((v.key, port, external) for v in layers for port, external in v.declaration["inputs"]),
            self.declaration["links"], self.declaration["outputs"])
        result.template()
        return result

    def solve(self, pins=(), arguments=None, sources=None):
        return LayerSolution.compute(self, pins, arguments, sources)


def _pins(system, pins):
    pins = tuple(tuple(row) for row in pins)
    if len(pins) > 65536:
        raise ValueError("layer observation limit")
    entries, values, keys = [], [], []
    for index, row in enumerate(pins):
        if len(row) != 3:
            raise ValueError("pin requires layer identity, coordinate and exact value")
        layer, coordinate, value = row
        action = system.selection(layer)
        position = action.codomain.keys.index(coordinate)
        entries.extend((index, j, v) for i, j, v in action.entries if i == position)
        keys.append(str(index)); values.append(exact_fraction(value))
    space = CoordinateSpace("layer observations", tuple(keys))
    return CoordinateMap(system.space, space, tuple(entries)), TensorField(space, values, source=system.source)


@dataclass(frozen=True)
class LayerSolution(_Value):
    data: bytes
    _kind = "layer_solution"

    def __post_init__(self):
        d = self.declaration
        if set(d) != {"product", "pins", "arguments", "status", "family", "native", "certificate"}:
            raise ValueError("invalid layer solution fields")
        LayerProduct(d["product"])
        if d["status"] not in {"compatible", "inconsistent", "incomplete"}:
            raise ValueError("invalid layer compatibility status")

    @property
    def declaration(self):
        return _unpack(self.data, self._kind)

    @property
    def product(self):
        return LayerProduct(self.declaration["product"])

    @property
    def dependencies(self):
        return self.product.dependencies

    @classmethod
    def compute(cls, product, pins=(), arguments=None, sources=None):
        from rexgraph.section_calculus import InconsistentSectionError
        from rexgraph.section_codec import pack_section
        pins = tuple(tuple(row) for row in pins)
        data = {"product": product.data, "pins": pins, "arguments": arguments,
                "status": "compatible", "family": None, "native": None, "certificate": None}
        if product.profile == "affine":
            if arguments:
                raise ValueError("affine solving does not execute target programs")
            system = product.system(sources)
            try:
                family = system.complete(*_pins(system, pins)) if pins else system.complete()
            except InconsistentSectionError as error:
                from rexgraph.field_codec import pack_field
                data.update(status="inconsistent", certificate={"witness": pack_field(error.witness, native=True),
                    "contradiction": error.contradiction})
            else:
                data["family"] = pack_section(family, native=True)
        elif product.profile == "c_workflows":
            if pins or arguments:
                raise ValueError("C products retain native declarations; observations require typed native operations")
            from .c_workflow import validate_product, CInterface
            graph=validate_product(product,execute=True,sources=sources)
            data["native"]=graph.data
            data["certificate"]={"profile":"complete-native-interface", "proofs":tuple(CInterface(v).declaration["proof"] for v in product.declaration["interfaces"])}
        else:
            if pins:
                raise ValueError("native profiles use their explicit native observations")
            model = TemporalModel(product.layers[0].declaration["native"])
            layers = []
            for index, layer in enumerate(product.layers):
                d = layer.declaration
                identity = d["object"]
                if d["native_program"]:
                    if not isinstance(arguments, Mapping):
                        raise ValueError("function operator layers require explicit role bindings")
                    model = model.apply(d["native_program"], dict(arguments))
                    identity = model.selected
                layers.append((index+1, identity, d["support"], d["scale"], d["shift"], d["role"]))
            result = model.product(product.profile, tuple(layers), product.declaration["transitions"], product.declaration["name"])
            info = result.info()
            data["native"] = result.data
            data["status"] = "incomplete" if not info["complete"] else "compatible" if info["compatible"] else "inconsistent"
        return cls(_pack(cls._kind, data))

    def verify(self, sources=None):
        fresh = self.compute(self.product, self.declaration["pins"], self.declaration["arguments"], sources)
        if fresh.data != self.data:
            raise ValueError("saved layer solution differs from exact replay")
        return True

    def family(self, sources=None):
        from rexgraph.section_codec import unpack_section
        from .result_codec import bind_sources
        if self.product.profile != "affine" or self.declaration["status"] != "compatible":
            raise ValueError("layer solution has no compatible affine family")
        self.verify(sources)
        system = self.product.system(sources)
        family = unpack_section(self.declaration["family"])
        ref = system.source
        return bind_sources(family, {(r.state_digest, r.record_id, r.version): r.source
                                     for r in (ref, *system.recipe.dependencies)})

    def native(self, sources=None):
        if self.declaration["native"] is None:
            raise ValueError("affine product has no native temporal result")
        self.verify(sources)
        if self.product.profile == "c_workflows":
            from .c_workflow import CWorkflow
            graph=CWorkflow.from_bytes(self.declaration["native"])
            graph.require_sources(sources);graph.native()
            return graph
        return TemporalModel(self.declaration["native"])

    def observe(self, action, sources=None):
        if not isinstance(action, CoordinateMap):
            raise TypeError("layer readout requires an explicit coordinate map")
        return self.family(sources).observe(action)

    def compile(self, sources, parameters):
        return LayerOperator.compile(self, sources, parameters)


def _bound_dependencies(product, sources):
    if sources is None:
        return product.dependencies
    bindings = _execution_sources(product, sources)
    result = []
    for expected in product.dependencies:
        for actual in bindings.values():
            ref = FieldSource(actual.value, actual.ref.record_id, actual.ref.record_version)
            if ref.matches(expected):
                result.append(expected.bind(actual.value)); break
    return tuple(result)


def _execution_sources(product, sources, system=None):
    from .binding import Binding, bind
    from .capabilities import SourcePolicy
    if not isinstance(sources, Mapping) or not all(isinstance(v, Binding) for v in sources.values()):
        raise TypeError("layer execution requires explicit source Bindings")
    result = dict(sources)
    policies = []
    for expected in product.dependencies:
        found = None
        for actual in result.values():
            try:
                candidate = FieldSource(actual.value, actual.ref.record_id, actual.ref.record_version)
                if candidate.matches(expected):
                    found = actual; break
            except (TypeError, ValueError):
                continue
        if found is None:
            raise ValueError("layer dependency requires its exact current source version")
        found.source.require("read"); policies.append(found.source.policy)
    policy = SourcePolicy.intersection(*(policies or [v.source.policy for v in result.values()])) if result else SourcePolicy.allow("read", "identity")
    result = {key: bind(key, value.value, SourcePolicy.intersection(value.source.policy, policy),
                        source_ref=value.ref, temporal=value.temporal) for key, value in result.items()}
    if system is not None:
        key = "layer_diagram_"+product.coefficient_digest
        if key in result:
            raise ValueError("layer diagram source name collision")
        result[key] = bind(key, system.source.source, policy)
    return result


@dataclass(frozen=True)
class LayerOperator(_Value):
    """A compiled declaration with replayed mathematical premises on every use."""
    data: bytes
    _kind = "layer_operator"

    def __post_init__(self):
        d = self.declaration
        if set(d) != {"solution", "product", "program"}:
            raise ValueError("invalid layer operator fields")
        LayerProduct(d["product"])
        if d["solution"] is not None:
            solution = LayerSolution(d["solution"])
            if solution.product.data != d["product"]:
                raise ValueError("operator solution belongs to another layer diagram")
        if d["program"] is not None:
            Program.from_bytes(d["program"])

    @property
    def declaration(self):
        return _unpack(self.data, self._kind)

    @property
    def product(self):
        return LayerProduct(self.declaration["product"])

    @property
    def dependencies(self):
        return self.product.dependencies

    @classmethod
    def compile(cls, value, sources, parameters):
        from .program_family import ProgramFamily
        if isinstance(value, (LayerProduct, LayerSolution)):
            product = value if isinstance(value, LayerProduct) else value.product
            if product.profile == "c_workflows":
                from .c_workflow import validate_product
                if parameters or len(product.declaration["interfaces"]) != 1:
                    raise ValueError("C compilation returns one complete native interface with no implicit selection")
                _execution_sources(product,sources)
                validate_product(product,execute=True,sources=sources)
                return cls(_pack(cls._kind,{"solution":None,"product":product.data,"program":None}))
            if product.profile == "functions":
                if isinstance(value, LayerSolution) and value.declaration["status"] != "compatible":
                    raise ValueError("function compilation requires a compatible retained solution")
                _execution_sources(product, sources)
                return cls(_pack(cls._kind, {"solution": None, "product": product.data, "program": None}))
        solution = value.solve(sources=sources) if isinstance(value, LayerProduct) else value
        if not isinstance(solution, LayerSolution) or solution.declaration["status"] != "compatible":
            raise ValueError("layer compilation requires a complete compatible solution")
        product = solution.product
        if product.profile != "affine":
            raise TypeError("finite family results retain their native family; select a qualified native worker explicitly")
        assembly = product.assembly()
        if assembly is None:
            raise ValueError("layer product has no executable local programs")
        system = product.system(sources)
        bindings = _execution_sources(product, sources, system)
        family = ProgramFamily(assembly, system.recipe, solution.family(sources))
        program = family.compile(bindings, parameters)
        return cls(_pack(cls._kind, {"solution": solution.data, "product": product.data, "program": program.to_bytes()}))

    def apply(self, sources, parameters):
        """Return the declared outputs of the validated operator."""
        result = self.execute(sources, parameters)
        return result.values if self.product.profile == "affine" else result

    def execute(self, sources, parameters):
        """Retain the complete local program execution for affine products."""
        product = self.product
        _execution_sources(product, sources)
        if product.profile == "c_workflows":
            from .c_workflow import validate_product, CInterface
            if parameters or len(product.declaration["interfaces"]) != 1:
                raise ValueError("C application retains the complete native interface")
            graph=validate_product(product,execute=True,sources=sources)
            return graph.value(CInterface(product.declaration["interfaces"][0]).declaration["proof"])
        if product.profile == "functions":
            result = product.solve(arguments=parameters, sources=sources)
            if result.declaration["status"] != "compatible":
                raise ValueError("function layer interfaces are incompatible on the declared cover")
            native = result.native(sources)
            return native.select(native.info()["output"], "function")
        solution = LayerSolution(self.declaration["solution"])
        fresh = self.compile(solution, sources, parameters)
        if fresh.data != self.data:
            raise ValueError("layer operator differs from its current exact compilation")
        from .program_operators import _program_executor
        program = Program.from_bytes(self.declaration["program"])
        return program.execute(_program_executor(_execution_sources(product, sources, product.system(sources)), parameters))


@dataclass(frozen=True)
class LayerReadout(_Value):
    """A retained observation of the complete affine family."""
    data: bytes
    _kind = "layer_readout"

    def __post_init__(self):
        d = self.declaration
        if set(d) != {"solution", "map"}:
            raise ValueError("invalid layer readout fields")
        solution = LayerSolution(d["solution"])
        action = _read_map(d["map"])
        if solution.product.profile != "affine" or action.domain != solution.product.system().space:
            raise ValueError("readout domain differs from the layer carrier")

    @property
    def declaration(self):
        return _unpack(self.data, self._kind)

    @property
    def solution(self):
        return LayerSolution(self.declaration["solution"])

    @property
    def dependencies(self):
        return self.solution.dependencies

    @classmethod
    def create(cls, solution, action):
        return cls(_pack(cls._kind, {"solution": solution.data, "map": _map(action)}))

    def image(self, sources=None):
        return self.solution.observe(_read_map(self.declaration["map"]), sources)

    def value(self, sources=None):
        return self.image(sources).value()


LAYER_VALUES = {v.__name__: v for v in (Layer, LayerInterface, LayerProduct, LayerSolution, LayerOperator, LayerReadout)}
