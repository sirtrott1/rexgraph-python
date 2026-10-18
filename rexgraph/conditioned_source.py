"""Declared raw byte uses, prospective four use sources and conditioned source codecs."""
from importlib import import_module

def _native():
    return import_module("rexgraph_cpp")

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256

from rexgraph.model_state import freeze_tree
from rexgraph.value_codec import pack_value


PROFILE = "rexgraph.observed-language-input.v1"
_KEYS = {"profile", "document", "encoding", "raw", "byte_cutoff", "uses",
         "namespace", "semantic_scale", "relation_key", "asset_key", "channel"}


def _positive(value):
    if type(value) is not int or not 0 < value < 1 << 64:
        raise ValueError("positive original native identity required")
    return value


@dataclass(frozen=True)
class ObservedLanguageInput:
    """Owning native source and asset with passive, declared input provenance."""
    source: object
    asset: object
    declaration: object

    def __post_init__(self):
        object.__setattr__(self, "declaration", freeze_tree(self.declaration))

    def receipt(self):
        return freeze_tree({"profile": PROFILE, "document": self.declaration["document"],
                            "byte_cutoff": self.declaration["byte_cutoff"],
                            "semantic_scale": self.declaration["semantic_scale"],
                            "encoding": self.declaration["encoding"],
                            "source_sha256": sha256(self.source.archive).hexdigest(),
                            "asset_sha256": sha256(self.asset.archive()).hexdigest(),
                            "authority": "passive-source-no-execution-grant"})


def observed_language_input(record, *, max_bytes=256*1024*1024, max_uses=256):
    """Declare ordered uses without requiring disjoint or contiguous spans.

    The byte cutoff is an accession boundary, not an annotation availability
    grant. Model execution still requires its own native availability context.
    """
    if (not isinstance(record, Mapping) or set(record) != _KEYS
            or record["profile"] != PROFILE):
        raise ValueError("closed observed byte/use declaration required")
    if (type(record["raw"]) is not bytes or type(record["byte_cutoff"]) is not int
            or not 0 < record["byte_cutoff"] <= len(record["raw"])
            or type(record["document"]) is not str or not record["document"]
            or len(record["document"].encode("utf-8")) > 512
            or type(record["encoding"]) is not str or not record["encoding"]):
        raise ValueError("original bytes, encoding, document and observed cut required")
    if (type(max_bytes) is not int or not 0 < max_bytes < 1 << 64
            or type(max_uses) is not int or not 0 < max_uses <= 4096
            or type(record["semantic_scale"]) is not int or not 0 <= record["semantic_scale"] < 1 << 16):
        raise ValueError("explicit semantic scale and positive source bounds required")
    for key in ("namespace", "relation_key", "asset_key", "channel"):
        _positive(record[key])
    raw = record["raw"][:record["byte_cutoff"]]
    if len(raw) > max_bytes:
        raise ValueError("observed bytes exceed their admission")
    uses = tuple(record["uses"])
    if not 0 < len(uses) <= max_uses:
        raise ValueError("bounded nonempty ordered use declaration required")
    declared = []; identities = set()
    for use in uses:
        if not isinstance(use, Mapping) or set(use) != {"key", "spans"}:
            raise ValueError("closed original use identity and span list required")
        key = _positive(use["key"])
        if key in identities:
            raise ValueError("distinct original use identities required")
        identities.add(key); spans = tuple(tuple(span) for span in use["spans"])
        if (not spans or len(spans) > max_uses
                or any(len(span) != 2 or any(type(i) is not int for i in span)
                       or not 0 <= span[0] <= span[1] <= len(raw) for span in spans)):
            raise ValueError("complete ordered spans within observed bytes required")
        declared.append({"key": key, "spans": spans})
    declaration = dict(record, raw=raw, uses=tuple(declared))
    provenance = pack_value({key: declaration[key] for key in sorted(_KEYS - {"raw", "uses"})})
    source = _native().NativeSource.from_declaration({
        "profile": "declared-role-source-v1", "namespace": record["namespace"],
        "semantic_scale": record["semantic_scale"], "max_grade": 1,
        "role_aspects": ("observed-use",),
        "types": ({"name": "declared-observed-use-group", "min_arity": 1, "max_arity": max_uses},),
        "witnesses": tuple(use["key"] for use in declared),
        "relations": ({"grade": 1, "key": record["relation_key"], "type": 0,
                       "occurrences": tuple({"key": i+1, "lower": use["key"], "aspects": (0,),
                                             "exists": True, "negative": False, "magnitude": 1}
                                            for i,use in enumerate(declared))},),
        "carriers": ({"key": 1, "kind": "explicit", "grade": 0, "coordinates": 1, "size": len(provenance)},),
        "fields": ({"key": 1, "carrier": 1,
                    "entries": tuple((i,value) for i,value in enumerate(provenance) if value), "support": None},),
        "metrics": ()}, max_bytes=max_bytes)
    asset = _native().NativeInputUseAsset(source.archive, {
        "profile": "rexgraph.relational-input-use-asset.v1", "key": record["asset_key"],
        "channel": record["channel"], "encoding": record["encoding"], "bytes": raw,
        "uses": tuple({"key": use["key"], "cells": ((0,use["key"]),(1,record["relation_key"])),
                       "occurrences": ((1,i+1),), "spans": use["spans"]} for i,use in enumerate(declared))},
        max_bytes=max_bytes)
    return ObservedLanguageInput(source, asset, freeze_tree(declaration))

from dataclasses import dataclass
from hashlib import sha256

from rexgraph.model_state import freeze_tree, thaw_tree
from rexgraph.value_codec import pack_value


ROLE_SOURCE_PROFILE = "rexgraph.four-use-prospective-role-source.v1"
PAIR_ORDER = tuple((head, dependent) for head in range(4) for dependent in range(4)
                   if head != dependent)
ROLE_ASPECTS = ("observed-use", "prospective-head", "prospective-dependent")


@dataclass(frozen=True)
class FourUseRoleSource:
    parent: object
    observed_source: object
    asset: object
    model_source: object
    interfaces: object

    def __post_init__(self):
        object.__setattr__(self, "interfaces", freeze_tree(self.interfaces))

    def receipt(self):
        return freeze_tree({"profile": ROLE_SOURCE_PROFILE,
                            "parent": self.parent.receipt(),
                            "observed_source_sha256": sha256(self.observed_source.archive).hexdigest(),
                            "asset_sha256": sha256(self.asset.archive()).hexdigest(),
                            "model_source_sha256": sha256(self.model_source.archive).hexdigest(),
                            "interfaces": self.interfaces,
                            "authority": "initial-source-scaffold-no-execution-or-acquisition-grant"})


def four_use_role_source(record, *, namespace, model_namespace, max_bytes=256*1024*1024):
    """Declare every candidate pair using raw uses alone, retaining the parent.

    Both new sources prospectively declare the same aspect/type signature.
    They are new initial constructions. The old L01 source is retained as an
    actual dependency; it has no implicit morphism into this changed signature.
    """
    parent = observed_language_input(record, max_bytes=max_bytes)
    if (type(namespace) is not int or type(model_namespace) is not int
            or not 0 < namespace < 1 << 64 or not 0 < model_namespace < 1 << 64
            or len({namespace, model_namespace, parent.declaration["namespace"]}) != 3):
        raise ValueError("distinct declared parent, initial-source and model namespaces required")
    uses = parent.declaration["uses"]
    if len(uses) != 4:
        raise ValueError("exactly four original observed uses required")
    group = parent.declaration["relation_key"]
    if group + 12 >= 1 << 64:
        raise ValueError("prospective pair relation identities exceed native width")
    words = tuple(use["key"] for use in uses)
    provenance = pack_value({"profile": ROLE_SOURCE_PROFILE,
                             "parent": thaw_tree(parent.receipt()),
                             "pair_order": PAIR_ORDER,
                             "roles_are": "prospective interfaces, not observed grammar"})
    context = {"grade": 1, "key": group, "type": 0,
               "occurrences": tuple({"key": index+1, "lower": key, "aspects": (0,),
                                     "exists": True, "negative": False, "magnitude": 1}
                                    for index, key in enumerate(words))}
    declaration = {"profile": "declared-role-source-v1",
                   "namespace": namespace,
                   "semantic_scale": parent.declaration["semantic_scale"], "max_grade": 1,
                   "role_aspects": ROLE_ASPECTS,
                   "types": ({"name": "declared-observed-use-group", "min_arity": 4, "max_arity": 4},
                             {"name": "prospective-directed-role-pair", "min_arity": 2, "max_arity": 2}),
                   "witnesses": words, "relations": (context,),
                   "carriers": ({"key": 1, "kind": "explicit", "grade": 0,
                                 "coordinates": 1, "size": len(provenance)},),
                   "fields": ({"key": 1, "carrier": 1, "support": None,
                               "entries": tuple((i, value) for i, value in enumerate(provenance) if value)},),
                   "metrics": ()}
    observed = _native().NativeSource.from_declaration(declaration, max_bytes=max_bytes)
    asset = _native().NativeInputUseAsset(observed.archive, {
        "profile": "rexgraph.relational-input-use-asset.v1",
        "key": parent.declaration["asset_key"], "channel": parent.declaration["channel"],
        "encoding": parent.declaration["encoding"], "bytes": parent.declaration["raw"],
        "uses": tuple({"key": use["key"], "cells": ((0, use["key"]), (1, group)),
                       "occurrences": ((1, index+1),), "spans": use["spans"]}
                      for index, use in enumerate(uses))}, max_bytes=max_bytes)
    pairs = []; interfaces = []
    for index, (head, dependent) in enumerate(PAIR_ORDER):
        key = group + index + 1; head_flag = 5+2*index; dependent_flag = head_flag+1
        pairs.append({"grade": 1, "key": key, "type": 1, "occurrences": (
            {"key": head_flag, "lower": words[head], "aspects": (1,),
             "exists": True, "negative": False, "magnitude": 1},
            {"key": dependent_flag, "lower": words[dependent], "aspects": (2,),
             "exists": True, "negative": False, "magnitude": 1})})
        interfaces.append({"pair": (head, dependent), "relation": key,
                           "head_use": words[head], "dependent_use": words[dependent],
                           "head_flag": head_flag, "dependent_flag": dependent_flag,
                           "role_directions": ("nsubj", "obj"),
                           "role_location": "dependent-flag auxiliary channels",
                           "parameter_order": int(head > dependent)})
    model = _native().NativeSource.from_declaration(dict(declaration, namespace=model_namespace, relations=(context, *pairs)), max_bytes=max_bytes)
    return FourUseRoleSource(parent, observed, asset, model, freeze_tree(tuple(interfaces)))

from collections.abc import Mapping
from dataclasses import dataclass, replace
from hashlib import sha256

from rexgraph.model_state import freeze_tree, thaw_tree
from rexgraph.value_codec import pack_value, unpack_value


SCHEMA = "rexgraph.conditioned-language-source.v2"
PROFILES = ("literal-context-v2", "supplied-morphology-v2")
DEPENDENCY_KINDS = ("literal", "structure", "predicate-grammar", "context",
                    "dictionary", "morphology")
REQUEST_KEYS = {"profile", "key", "revision", "pattern_sha256", "dependencies",
                "cutoffs", "literal_dependency", "structure_dependency",
                "predicate_dependency", "predicates", "context_links", "symbolic", "morphology"}
DEPENDENCY_KEYS = {"key", "kind", "producer", "version", "origin", "namespace",
                   "sequence", "payload_sha256"}
UPOS = frozenset(("ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN",
                  "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"))


def _closed(value, keys):
    if not isinstance(value, Mapping) or set(value) != set(keys):
        raise ValueError("closed versioned declaration required")
    return value


def _text(value):
    if type(value) is not str or not value or len(value.encode("utf-8")) > 512:
        raise ValueError("bounded nonempty original identity required")
    return value


def _digest(value):
    if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("lowercase SHA256 required")
    return value


def _natural(value, *, zero=False):
    if type(value) is not int or not (0 if zero else 1) <= value < 1 << 64:
        raise ValueError("native unsigned identity required")
    return value


def _rows(values, maximum):
    if not isinstance(values, (tuple, list)) or len(values) > maximum:
        raise ValueError("bounded explicit ordered records required")
    return tuple(values)


def _sha(raw):
    return sha256(raw).hexdigest()


def _check_observed(value, maximum):
    if not isinstance(value, ObservedLanguageInput):
        raise TypeError("actual observed byte/use owner required")
    replay = observed_language_input(value.declaration, max_bytes=maximum)
    if (replay.source.archive != value.source.archive
            or replay.asset.archive() != value.asset.archive()
            or value.asset.source_archive() != value.source.archive):
        raise ValueError("observed declaration differs from native byte/use owner")
    if value.declaration["encoding"] != "UTF-8":
        raise ValueError("v2 initial profile requires explicit UTF-8")
    raw = value.declaration["raw"]
    try:
        raw.decode("utf-8")
        for use in value.declaration["uses"]:
            for begin, end in use["spans"]:
                raw[:begin].decode("utf-8")
                raw[begin:end].decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("UTF-8 use cuts must preserve character boundaries") from error


def _symbolic_observation(row):
    _closed(row, ("state", "values"))
    if row["state"] not in ("observed", "alternatives", "missing"):
        raise ValueError("explicit observed, alternative or missing state required")
    values = _rows(row["values"], 256)
    if (any(type(v) is not str or len(v.encode()) > 512 for v in values)
            or len(set(values)) != len(values)
            or (row["state"] == "observed" and len(values) != 1)
            or (row["state"] == "alternatives" and len(values) < 2)
            or (row["state"] == "missing" and values)):
        raise ValueError("observation state differs from retained symbolic alternatives")
    return {"state": row["state"], "values": values}


def conditioned_port_schema():
    """Declare source domains; these descriptions contain no executable maps."""
    return freeze_tree({
        "literal": {"owner": "NativeInputUseAsset", "domain": "original ordered byte spans",
                    "accession": "native recover(use), preserving each span separately",
                    "reconstruction": "whole retained byte asset; span accession alone has no inverse",
                    "metric": None, "image": "actual bytes at supplied cuts", "residual": "whole byte asset retained"},
        "conditions": {"domain": "joint finite predicate/context alternatives",
                       "states": ("observed-false", "observed-true", "unresolved", "missing"),
                       "vectorization": None, "admission": "native actual asset/use/predicate/origin owner required"},
        "literal_observations": {"owner": "native source-derived literal input, A01 admission required",
                                 "basis": "ordered (original-use, original-span-index, predicate-key) directions",
                                 "domain": "complete supported Boolean observations in rational stalks",
                                 "metric": "positive identity on declared native stalks",
                                 "A": "native declared complete accession", "R": "native declared reconstruction",
                                 "image": "native source-derived predicate image; not arbitrary caller truth bits",
                                 "residual": "retain or reject explicitly; no inferred zero outside image"},
        "linear_input": {"location": "each original structural use", "basis": ("declared-use-unit",),
                         "domain": "rational", "metric": "identity", "A": "declared identity",
                         "R": "declared identity", "image": "whole named stalk", "residual": "zero only for identity frame"},
        "linear_response": {"location": "each prospective dependent flag in PAIR_ORDER",
                            "basis": ("prospective-nsubj", "prospective-obj"), "domain": "rational",
                            "metric": "identity", "A": "declared identity", "R": "declared identity",
                            "image": "whole named stalk; terminal Boolean grammar separately admitted",
                            "residual": "zero only for identity frame"},
        "sharing": {"pattern": "prospective-directed-role-pair", "instances": PAIR_ORDER,
                    "maps": "explicit use/flag instance maps; no equality-of-values inference",
                    "complete_action": "R_tau W_pattern A_sigma + K (I - R_sigma A_sigma)",
                    "complement": "identity frames initially; changed frames require complete K and residual contract",
                    "differential": "original parent retained; native A02 determines allowed learned realization and both adjacent laws"},
        "status": "schema-only; native A01/A02 admission and action review required"})


@dataclass(frozen=True)
class ConditionedLanguageSource:
    """Passive v2 declaration retaining actual original native owners."""
    parent: object
    contexts: tuple
    symbolic: tuple
    declaration: object

    def __post_init__(self):
        object.__setattr__(self, "contexts", tuple(self.contexts))
        object.__setattr__(self, "symbolic", tuple(self.symbolic))
        object.__setattr__(self, "declaration", freeze_tree(self.declaration))

    def archive(self):
        return pack_value(thaw_tree(self.declaration))

    def receipt(self):
        return freeze_tree({"schema": SCHEMA, "profile": self.declaration["identity"]["profile"],
                            "declaration_sha256": _sha(self.archive()),
                            "parent_source_sha256": _sha(self.parent.model_source.archive),
                            "literal_asset_sha256": _sha(self.parent.asset.archive()),
                            "authority": "declaration-only-no-execution-or-availability-grant"})


def conditioned_language_source(parent, request, *, contexts=None, symbolic=None,
                                max_bytes=16*1024*1024, max_records=4096):
    """Validate a target free schema, without evaluating predicates or fields."""
    _closed(request, REQUEST_KEYS)
    if (type(max_bytes) is not int or not 0 < max_bytes <= 256*1024*1024
            or type(max_records) is not int or not 0 < max_records <= 4096):
        raise ValueError("positive bounded source admissions required")
    if not isinstance(parent, FourUseRoleSource):
        raise TypeError("actual original four-use parent required")
    _check_observed(parent.parent, max_bytes)
    replay = four_use_role_source(parent.parent.declaration,
        namespace=parent.observed_source.identity["namespace"],
        model_namespace=parent.model_source.identity["namespace"], max_bytes=max_bytes)
    if (replay.observed_source.archive != parent.observed_source.archive
            or replay.model_source.archive != parent.model_source.archive
            or replay.asset.archive() != parent.asset.archive()
            or thaw_tree(replay.interfaces) != thaw_tree(parent.interfaces)):
        raise ValueError("parent source, flags, byte owner or interfaces differ")
    if request["profile"] not in PROFILES:
        raise ValueError("explicit v2 literal or supplied-morphology profile required")
    _natural(request["key"]); _natural(request["revision"]); _digest(request["pattern_sha256"])
    contexts = {} if contexts is None else contexts
    symbolic = {} if symbolic is None else symbolic
    if (not isinstance(contexts, Mapping) or not isinstance(symbolic, Mapping)
            or len(contexts) > 16 or len(symbolic) > 16):
        raise ValueError("bounded named actual context/symbolic owners required")
    for key, value in contexts.items():
        _text(key); _check_observed(value, max_bytes)
    cutoffs = {}
    for row in _rows(request["cutoffs"], max_records):
        _closed(row, ("namespace", "sequence"))
        namespace = _natural(row["namespace"]); _natural(row["sequence"], zero=True)
        if namespace in cutoffs:
            raise ValueError("one publication cutoff per publisher required")
        cutoffs[namespace] = row["sequence"]
    dependencies = {}
    for row in _rows(request["dependencies"], max_records):
        _closed(row, DEPENDENCY_KEYS)
        key = _text(row["key"])
        if key in dependencies or row["kind"] not in DEPENDENCY_KINDS:
            raise ValueError("unique typed dependency required")
        for name in ("producer", "version", "origin"):
            _text(row[name])
        _digest(row["payload_sha256"]); _natural(row["namespace"]); _natural(row["sequence"], zero=True)
        if row["namespace"] not in cutoffs or row["sequence"] > cutoffs[row["namespace"]]:
            raise ValueError("dependency is outside its declared publication cutoff")
        dependencies[key] = dict(row)

    used_dependencies = set()

    def dependency(key, kind, payload=None):
        _text(key)
        row = dependencies.get(key)
        if row is None or row["kind"] != kind or (payload is not None and row["payload_sha256"] != _sha(payload)):
            raise ValueError("dependency kind, payload or original identity differs")
        used_dependencies.add(key)
        return key

    dependency(request["literal_dependency"], "literal", parent.asset.archive())
    dependency(request["structure_dependency"], "structure", parent.model_source.archive)
    predicates = []; keys = set()
    for row in _rows(request["predicates"], max_records):
        _closed(row, ("key", "kind", "argument"))
        key = _natural(row["key"])
        if (key in keys or row["kind"] not in ("byte-equal", "byte-prefix", "byte-suffix", "byte-punctuation")
                or type(row["argument"]) is not bytes or len(row["argument"]) > 256):
            raise ValueError("closed exact byte predicate grammar required; no supplied results")
        if row["kind"] == "byte-punctuation" and row["argument"]:
            raise ValueError("native ASCII punctuation predicate has no operand")
        keys.add(key); predicates.append(dict(row))
    if not predicates:
        raise ValueError("nonempty frozen literal predicate grammar required")
    dependency(request["predicate_dependency"], "predicate-grammar", pack_value(predicates))
    local_uses = {u["key"] for u in parent.parent.declaration["uses"]}
    links = []; linked_contexts = set(); link_keys = set()
    for row in _rows(request["context_links"], max_records):
        _closed(row, ("key", "local_use", "context", "context_use", "dependency"))
        if (_text(row["key"]) in link_keys or row["local_use"] not in local_uses
                or row["context"] not in contexts):
            raise ValueError("context link must retain actual local and external use owners")
        owner = contexts[row["context"]]
        if type(row["local_use"]) is not int or type(row["context_use"]) is not int or row["context_use"] not in {u["key"] for u in owner.declaration["uses"]}:
            raise ValueError("context link references another original use")
        dependency(row["dependency"], "context", owner.asset.archive())
        links.append(dict(row)); linked_contexts.add(row["context"]); link_keys.add(row["key"])
    if linked_contexts != set(contexts):
        raise ValueError("every supplied context asset requires an explicit use interface")
    symbolic_rows = []; symbolic_keys = set()
    for row in _rows(request["symbolic"], 16):
        _closed(row, ("key", "asset", "dependency"))
        key = _text(row["key"])
        if key in symbolic_keys or key not in symbolic:
            raise ValueError("unique actual symbolic owner required")
        owner = symbolic[key]
        asset = parent.asset if row["asset"] == "literal" else contexts.get(row["asset"])
        if isinstance(asset, ObservedLanguageInput):
            asset = asset.asset
        if not isinstance(owner, _native().NativeFiniteSource) or asset is None or not owner.asset().same_owner(asset):
            raise ValueError("symbolic alternatives require their actual original asset owner")
        dependency(row["dependency"], "dictionary", owner.archive())
        symbolic_rows.append(dict(row, archive=owner.archive())); symbolic_keys.add(key)
    if symbolic_keys != set(symbolic):
        raise ValueError("all symbolic owners require declared interfaces")
    morphology = []; seen = set()
    for row in _rows(request["morphology"], 4):
        _closed(row, ("use", "upos", "xpos", "feats", "dependency"))
        if type(row["use"]) is not int or row["use"] not in local_uses or row["use"] in seen:
            raise ValueError("one supplied morphology record per original use required")
        values = {name: _symbolic_observation(row[name]) for name in ("upos", "xpos", "feats")}
        if any(v not in UPOS for v in values["upos"]["values"]):
            raise ValueError("UPOS alternatives require the declared tag vocabulary")
        dependency(row["dependency"], "morphology", pack_value({"use": row["use"], **values}))
        morphology.append({"use": row["use"], **values, "dependency": row["dependency"]}); seen.add(row["use"])
    if request["profile"] == "literal-context-v2":
        if morphology or any(row["kind"] == "morphology" for row in dependencies.values()):
            raise ValueError("literal profile cannot carry teacher morphology")
    elif seen != local_uses:
        raise ValueError("supplied profile explicitly records UPOS/XPOS/FEATS for all four uses")
    if used_dependencies != set(dependencies):
        raise ValueError("every declared dependency requires an actual input or observation interface")
    declaration = {
        "schema": SCHEMA,
        "identity": {"profile": request["profile"], "key": request["key"], "revision": request["revision"],
                     "document": parent.parent.declaration["document"], "pattern_sha256": request["pattern_sha256"]},
        "literal": {"parent_record": thaw_tree(parent.parent.declaration), "asset_archive": parent.asset.archive()},
        "structure": {"original_parent_archive": parent.parent.source.archive,
                      "observed_archive": parent.observed_source.archive, "model_archive": parent.model_source.archive,
                      "coordinate_order": parent.model_source.identity["coordinates"],
                      "semantic_scale": parent.parent.declaration["semantic_scale"],
                      "parent_projection": "identity on unchanged model source at every retained grade; original v1 packets retained"},
        "observations": {"predicates": predicates, "predicate_results": "native-admission-pending",
                         "morphology": morphology, "symbolic": symbolic_rows,
                         "contexts": tuple({"key": key, "record": thaw_tree(value.declaration),
                                            "asset_archive": value.asset.archive()} for key, value in contexts.items())},
        "interfaces": {"pairs": thaw_tree(parent.interfaces), "literal_uses": parent.asset.uses(max_records=4),
                       "context_links": links, "ports": thaw_tree(conditioned_port_schema())},
        "representations": {"frames": "identity on declared initial linear stalks; no byte-to-latent encoding",
                            "predicate_scope": "each span separately, ordered span results retained; no implicit concatenation",
                            "native_proof_required": True, "availability_grant": False},
        "availability": {"dependencies": tuple(dependencies.values()), "cutoffs": request["cutoffs"]},
        "policy": {"max_bytes": max_bytes, "max_records": max_records, "uses": 4,
                   "source_differential": "original parent retained; parameterized child realization requires native contract",
                   "teacher_grammar_switch": "unchanged v1 Gamma68 only"},
        "request": thaw_tree(request)}
    if len(pack_value(thaw_tree(declaration))) > max_bytes:
        raise ValueError("conditioned source packet exceeds its admission")
    return ConditionedLanguageSource(parent, tuple(contexts.items()), tuple(symbolic.items()), declaration)


def restore_conditioned_language_source(archive, *, max_bytes=16*1024*1024, max_records=4096):
    """Rebuild native data owners; restoring this packet grants no execution."""
    if type(archive) is not bytes or not 0 < len(archive) <= max_bytes:
        raise ValueError("bounded immutable source packet required")
    row = unpack_value(archive)
    _closed(row, ("schema", "identity", "literal", "structure", "observations", "interfaces",
                  "representations", "availability", "policy", "request"))
    if row["schema"] != SCHEMA:
        raise ValueError("unknown conditioned source schema")

    parent = four_use_role_source(row["literal"]["parent_record"],
        namespace=_native().NativeSource(row["structure"]["observed_archive"], max_bytes=max_bytes).identity["namespace"],
        model_namespace=_native().NativeSource(row["structure"]["model_archive"], max_bytes=max_bytes).identity["namespace"], max_bytes=max_bytes)
    contexts = {c["key"]: observed_language_input(c["record"], max_bytes=max_bytes)
                for c in row["observations"]["contexts"]}
    symbolic = {}; adopted = set()
    for item in row["observations"]["symbolic"]:
        asset_name = item["asset"]
        if asset_name in adopted:
            raise ValueError("multiple symbolic sources per asset require native common-owner bundle admission")
        finite = _native().NativeFiniteSource.from_archive(item["archive"], max_bytes=max_bytes)
        asset = finite.asset()
        expected = parent.asset if asset_name == "literal" else contexts[asset_name].asset
        if asset.archive() != expected.archive():
            raise ValueError("restored finite source differs from saved original byte/use asset")
        # Adopt the actual owner retained inside the restored finite source.
        # No ownership identity is inferred for the separately decoded asset.
        if asset_name == "literal":
            parent = replace(parent, asset=asset)
        else:
            contexts[asset_name] = replace(contexts[asset_name], asset=asset)
        symbolic[item["key"]] = finite; adopted.add(asset_name)
    policy = row["policy"]
    _closed(policy, ("max_bytes", "max_records", "uses", "source_differential", "teacher_grammar_switch"))
    if (type(policy["max_bytes"]) is not int or not 0 < policy["max_bytes"] <= max_bytes
            or type(policy["max_records"]) is not int or not 0 < policy["max_records"] <= max_records):
        raise ValueError("recorded source policy exceeds restoration admission")
    result = conditioned_language_source(parent, row["request"], contexts=contexts, symbolic=symbolic,
                                         max_bytes=policy["max_bytes"], max_records=policy["max_records"])
    if result.archive() != archive:
        raise ValueError("restored declaration differs from original source/interface packets")
    return result
