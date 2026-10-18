"""Exact certificates for declared relations between original transport entries."""
from collections import OrderedDict
from collections.abc import Mapping
from fractions import Fraction
from dataclasses import dataclass
from threading import RLock

import numpy as np

from rexgraph.cpp_stalk_fields import _exact
from rexgraph.cpp_stalk_transport import _prepare, _receipt_matches
from rexgraph.model_state import ModelOutput, ModelState, freeze_tree
from rexgraph.type_accession import CoordinateSpace

_QUERY = "whole-stalk-parameter-relations-query-v1"
_CONTEXT_QUERY = "whole-stalk-context-hypotheses-query-v1"
_VIEWS_QUERY = "whole-stalk-context-views-query-v1"
_DIAGRAMS_QUERY = "whole-stalk-context-diagrams-query-v1"
_RULES_QUERY = "whole-stalk-context-rules-query-v1"
_VIEW_RULES_QUERY = "whole-stalk-context-view-rules-query-v1"
_OBSERVED_VIEW_RULES_QUERY = "whole-stalk-observed-context-view-rules-query-v1"
_TEMPLATE_VIEW_RULES_QUERY = "whole-stalk-declared-template-view-rules-query-v1"
_VIEW_RULE_QUERIES = (_VIEW_RULES_QUERY, _OBSERVED_VIEW_RULES_QUERY, _TEMPLATE_VIEW_RULES_QUERY)
_RULE_QUERIES = (_RULES_QUERY, *_VIEW_RULE_QUERIES)
_CONTEXT_QUERIES = (_CONTEXT_QUERY, _VIEWS_QUERY, _DIAGRAMS_QUERY, *_RULE_QUERIES)
_METHOD = "exact-stalk-parameter-relations-v1"
_lock = RLock()
_cache = OrderedDict()


def _relations(state, relations):
    bounds = state.configuration["limits"]
    slots = state.configuration["parameter_slots"]
    records = []; names = set(); count = 0
    for name, terms, expected in relations:
        if (type(name) is not str or not name.strip() or len(name.encode()) > 256 or
                name in names or len(records) >= bounds[0]):
            raise ValueError("distinct bounded parameter relation names required")
        names.add(name); count += len(name.encode())+2; support = set(); row = []
        for entry, value in terms:
            count += 2
            if (type(entry) is not int or not 0 <= entry < len(slots) or entry in support or count > bounds[1]):
                raise ValueError("distinct admitted original variable map entries required")
            support.add(entry); row.append((entry, _exact(value, bounds[3])))
        if not row:
            raise ValueError("nonempty supported parameter relation required")
        records.append((name, tuple(row), _exact(expected, bounds[3])))
    if not records or count > bounds[1]:
        raise ValueError("admitted nonempty parameter relation family required")
    return tuple(records)


def _selected(relations, selected):
    if isinstance(selected, (str, bytes)):
        raise TypeError("selected relations require a sequence of names")
    names = {row[0] for row in relations}; result = []
    for name in selected:
        if type(name) is not str or name not in names or name in result:
            raise ValueError("distinct selected original parameter relation names required")
        result.append(name)
    return tuple(result)


def stalk_parameter_relations_query(state, relations, *, selected=()):
    """Declare linear entry equations and an optional joint conditional family."""
    state.check_state()
    if state.adapter != "cpp_stalk_transport":
        raise ValueError("original observed transport model required")
    records = _relations(state, relations)
    return dict(profile=_QUERY, source_reference=state.source.as_record(), relations=records,
                selected=_selected(records, selected))


def _query(state, query):
    if isinstance(query, Mapping) and query.get("profile") in _RULE_QUERIES:
        if (set(query) != {"profile", "source_reference", "training", "targets", "frames", "minimum_peers"} or
                not isinstance(query["source_reference"], Mapping) or
                not _receipt_matches(state.source.as_record(), query["source_reference"])):
            raise ValueError("closed context rule query bound to its original source required")
        declare = (stalk_declared_context_view_rules_query if query["profile"] == _TEMPLATE_VIEW_RULES_QUERY else
            stalk_context_view_rules_query if query["profile"] in _VIEW_RULE_QUERIES else stalk_context_rules_query)
        options = (dict(observed_inputs=query["profile"] == _OBSERVED_VIEW_RULES_QUERY)
            if query["profile"] in (_VIEW_RULES_QUERY, _OBSERVED_VIEW_RULES_QUERY) else {})
        return freeze_tree(declare(state, query["training"], query["targets"],
            query["frames"], minimum_peers=query["minimum_peers"], **options))
    if isinstance(query, Mapping) and query.get("profile") == _DIAGRAMS_QUERY:
        if (set(query) != {"profile", "source_reference", "target_patches", "frames", "selected_diagrams"} or
                not isinstance(query["source_reference"], Mapping) or
                not _receipt_matches(state.source.as_record(), query["source_reference"])):
            raise ValueError("closed context diagram query bound to its original source required")
        return freeze_tree(stalk_context_diagrams_query(state, query["target_patches"], query["frames"],
            selected_diagrams=query["selected_diagrams"]))
    if isinstance(query, Mapping) and query.get("profile") in (_CONTEXT_QUERY, _VIEWS_QUERY):
        if (set(query) != {"profile", "source_reference", "targets", "frames", "selected_blocks"} or
                not isinstance(query["source_reference"], Mapping) or
                not _receipt_matches(state.source.as_record(), query["source_reference"])):
            raise ValueError("closed contextual hypothesis query bound to its original source required")
        declare = stalk_context_views_query if query["profile"] == _VIEWS_QUERY else stalk_context_query
        return freeze_tree(declare(state, query["targets"], query["frames"],
            selected_blocks=query["selected_blocks"]))
    if (not isinstance(query, Mapping) or set(query) != {"profile", "source_reference", "relations", "selected"} or
            query["profile"] != _QUERY or not isinstance(query["source_reference"], Mapping) or
            not _receipt_matches(state.source.as_record(), query["source_reference"])):
        raise ValueError("closed parameter relation query bound to the actual transport source required")
    records = _relations(state, query["relations"])
    return freeze_tree(dict(profile=_QUERY, source_reference=state.source.as_record(), relations=records,
                            selected=_selected(records, query["selected"])))


def stalk_context_query(state, targets, frames, *, selected_blocks=()):
    """Acquire typed star hypotheses with explicitly declared stalk frame maps."""
    state.check_state()
    if state.adapter != "cpp_stalk_transport":
        raise ValueError("original observed transport model required")
    axes = state.configuration["stalk_axes"]; bounds = state.configuration["limits"]
    if isinstance(targets, (str, bytes)) or isinstance(frames, (str, bytes)) or len(frames) != 2:
        raise ValueError("original target rows and complete two-grade frame atlas required")
    rows = []; atlas = []; count = 0
    for row in targets:
        if type(row) is not int or not 0 <= row < len(axes[1]) or row in rows or len(rows) >= bounds[0]:
            raise ValueError("distinct admitted original target rows required")
        rows.append(row); count += 1
    for k, family in enumerate(frames):
        if isinstance(family, (str, bytes)) or len(family) != len(axes[k]):
            raise ValueError("frame atlas must cover every original stalk")
        records = []
        for i, frame in enumerate(family):
            count += 2
            if frame is None:
                records.append(None); continue
            name, permutation = frame
            if type(name) is not str or not name.strip() or len(name.encode()) > 256 or isinstance(permutation, (str, bytes)):
                raise ValueError("bounded common frame name and explicit permutation required")
            values = []
            for axis in permutation:
                count += 1
                if type(axis) is not int or not 0 <= axis < len(axes[k][i]) or axis in values or count > bounds[1]:
                    raise ValueError("complete admitted frame permutation required")
                values.append(axis)
            if len(values) != len(axes[k][i]):
                raise ValueError("frame map must include every original stalk direction")
            count += len(name.encode()); records.append((name, tuple(values)))
        atlas.append(tuple(records))
    selected = []
    if isinstance(selected_blocks, (str, bytes)):
        raise TypeError("selected context blocks require original row pairs")
    for origin, target in selected_blocks:
        pair = (origin, target); count += 2
        if (type(origin) is not int or type(target) is not int or not 0 <= origin < len(axes[1]) or
                target not in rows or pair in selected or len(selected) >= bounds[0]):
            raise ValueError("distinct selected original context row pairs required")
        selected.append(pair)
    if count > bounds[1]:
        raise ValueError("context declaration exceeds entry admission")
    return dict(profile=_CONTEXT_QUERY, source_reference=state.source.as_record(), targets=tuple(rows),
                frames=tuple(atlas), selected_blocks=tuple(selected))


def stalk_context_views_query(state, targets, frames, *, selected_blocks=()):
    """Declare faithful rational views and compare their actual common input image."""
    from rexgraph_cpp._context_views import declaration
    state.check_state()
    if state.adapter != "cpp_stalk_transport":
        raise ValueError("original observed transport model required")
    records = declaration(state.configuration["stalk_axes"], state.configuration["limits"],
        targets, frames, selected_blocks)
    return dict(profile=_VIEWS_QUERY, source_reference=state.source.as_record(), **records)


def stalk_context_rules_query(state, training, targets, frames, *, minimum_peers=2):
    """Learn conditional context sharing from independent observed original entries."""
    base = stalk_context_query(state, targets, frames)
    return _rule_query(state, training, base, minimum_peers, _RULES_QUERY)


def stalk_context_view_rules_query(state, training, targets, frames, *, minimum_peers=2, observed_inputs=False):
    """Learn conditional rules on shared inputs covered by independent exact peers."""
    base = stalk_context_views_query(state, targets, frames)
    if type(observed_inputs) is not bool:
        raise TypeError("observed input scope must be Boolean")
    return _rule_query(state, training, base, minimum_peers,
        _OBSERVED_VIEW_RULES_QUERY if observed_inputs else _VIEW_RULES_QUERY)


def stalk_declared_context_view_rules_query(state, training, targets, frames, *, minimum_peers=2):
    """Compare original stars within an explicitly named incidence template family."""
    base = stalk_context_views_query(state, targets, frames)
    return _rule_query(state, training, base, minimum_peers, _TEMPLATE_VIEW_RULES_QUERY)


def _rule_query(state, training, base, minimum_peers, profile):
    if "parameter_space" in state.configuration and profile != _TEMPLATE_VIEW_RULES_QUERY:
        raise ValueError("context rules require independent original map entries")
    if profile == _TEMPLATE_VIEW_RULES_QUERY and "parameter_space" not in state.configuration:
        raise ValueError("declared context templates require a named parameter basis")
    if type(minimum_peers) is not int or minimum_peers <= 0:
        raise ValueError("positive independent peer admission required")
    if isinstance(training, (str, bytes)):
        raise TypeError("original training row sequence required")
    rows = []
    for row in training:
        if (type(row) is not int or not 0 <= row < len(state.configuration["stalk_axes"][1]) or
                row in rows or row in base["targets"] or len(rows) >= state.configuration["limits"][0]):
            raise ValueError("distinct admitted training rows outside the target family required")
        rows.append(row)
    return dict(profile=profile, source_reference=base["source_reference"], training=tuple(rows),
        targets=base["targets"], frames=base["frames"], minimum_peers=minimum_peers)


def stalk_context_diagrams_query(state, target_patches, frames, *, selected_diagrams=()):
    """Declare connected original patches and complete conditional row maps."""
    from rexgraph_cpp._context_views import diagram_declaration
    state.check_state()
    if state.adapter != "cpp_stalk_transport":
        raise ValueError("original observed transport model required")
    records = diagram_declaration(state.configuration["stalk_axes"], state.configuration["limits"],
        target_patches, frames, selected_diagrams)
    return dict(profile=_DIAGRAMS_QUERY, source_reference=state.source.as_record(), **records)


def _context_analysis(state, transport, query):
    from rexgraph_cpp import PreparedStalkParameterRelations
    diagrams = query["profile"] == _DIAGRAMS_QUERY
    view_rules = query["profile"] in _VIEW_RULE_QUERIES
    rules = query["profile"] in _RULE_QUERIES
    views = query["profile"] in (_VIEWS_QUERY, _DIAGRAMS_QUERY, *_VIEW_RULE_QUERIES)
    targets = query["target_patches"] if diagrams else query["targets"]
    key = (state._digest, query["profile"], targets, query["frames"],
        (query["training"], query["minimum_peers"]) if rules else None)
    with _lock:
        cached = _cache.get(key)
        if cached is None:
            if view_rules:
                acquisition, plan = transport.contextual_view_rules(query["training"], targets, query["frames"],
                    minimum_peers=query["minimum_peers"], observed_inputs=query["profile"] == _OBSERVED_VIEW_RULES_QUERY,
                    declared_templates=query["profile"] == _TEMPLATE_VIEW_RULES_QUERY)
            elif rules:
                acquisition = transport.contextual_rules(query["training"], targets, query["frames"],
                    minimum_peers=query["minimum_peers"])
            else:
                discover = transport.contextual_diagrams if diagrams else transport.contextual_views if views else transport.contextual_hypotheses
                acquisition = discover(targets, query["frames"])
            relations = acquisition["relations"]
            if not view_rules:
                plan = PreparedStalkParameterRelations(transport, relations, allow_constants=views) if relations else None
            steps = acquisition["execution_steps"]
            blocks = []
            groups = []
            if plan is not None:
                certificate = plan.receipt; steps += certificate["preparation_steps"]
                for index, block in enumerate(acquisition["blocks"]):
                    if view_rules:
                        status = acquisition["block_statuses"][index]
                        blocks.append(dict(block, status=status, joint_consistent=status != "excluded",
                            joint_rank=acquisition["block_ranks"][index]))
                        continue
                    names = tuple(relations[i][0] for i in block["equations"])
                    forced = all(certificate["statuses"][i] == "forced" for i in block["equations"])
                    family = None if forced and rules else plan.condition(names)
                    if family is not None:
                        steps += family["execution_steps"]
                    if steps > state.configuration["limits"][2]:
                        raise ValueError("complete context acquisition exceeds work admission")
                    status = "forced" if forced else "excluded" if not family["consistent"] else "open"
                    blocks.append(dict(block, status=status, joint_consistent=True if forced else family["consistent"],
                        joint_rank=0 if family is None else family["rank"]))
                if diagrams:
                    for diagram in acquisition["diagrams"]:
                        names = tuple(relations[i][0] for i in diagram["equations"])
                        family = plan.condition(names); steps += family["execution_steps"]
                        if steps > state.configuration["limits"][2]:
                            raise ValueError("complete diagram acquisition exceeds work admission")
                        status = ("excluded" if not family["consistent"] else
                            "forced" if all(certificate["statuses"][i] == "forced" for i in diagram["equations"]) else "open")
                        groups.append(dict(diagram, status=status, joint_consistent=family["consistent"], joint_rank=family["rank"]))
            acquisition.update(blocks=tuple(blocks), certificate_steps=steps,
                evidence_scope="independent-original-entries" if "parameter_space" not in state.configuration else
                    "conditional-on-declared-parameter-basis",
                interpretation="complete-context-common-input-hypotheses-v1" if views else "complete-context-block-hypotheses-v1")
            if diagrams:
                acquisition.update(diagrams=tuple(groups), interpretation="connected-context-diagram-hypotheses-v1")
            if rules:
                if tuple(b["status"] for b in blocks) != acquisition["block_statuses"]:
                    raise ValueError("learned rule classifications differ from independent evidence replay")
                acquisition.update(interpretation="declared-template-context-view-rules-v1" if query["profile"] == _TEMPLATE_VIEW_RULES_QUERY else
                    "learned-observed-context-view-rules-v1" if query["profile"] == _OBSERVED_VIEW_RULES_QUERY else
                    "learned-covered-context-view-rules-v1" if view_rules else
                    "learned-conditional-context-rules-v1")
            cached = (plan, freeze_tree(acquisition)); _cache[key] = cached
            while len(_cache) > 4:
                _cache.popitem(last=False)
        _cache.move_to_end(key)
    plan, acquisition = cached
    selected = set(acquisition["selected_blocks"] if rules else
        query["selected_diagrams"] if diagrams else query["selected_blocks"]); names = []; found = set()
    for block in acquisition["diagrams"] if diagrams else acquisition["blocks"]:
        pair = tuple(block["row_map"]) if diagrams else (block["origin"], block["target"])
        if pair in selected:
            found.add(pair); names.extend(acquisition["relations"][i][0] for i in block["equations"])
    if found != selected:
        raise ValueError("selected context block has no acquired complete typed correspondence")
    return plan, acquisition, tuple(dict.fromkeys(names))


def _analysis(state, source, query):
    from rexgraph_cpp import PreparedStalkParameterRelations
    transport = _prepare(state, source); q = _query(state, query)
    acquisition = None
    if q["profile"] in _CONTEXT_QUERIES:
        plan, acquisition, chosen = _context_analysis(state, transport, q)
        relations = acquisition["relations"]
    else:
        key = (state._digest, tuple((name, tuple((i, v.numerator, v.denominator) for i, v in terms),
            expected.numerator, expected.denominator) for name, terms, expected in q["relations"]))
        with _lock:
            plan = _cache.get(key)
            if plan is None:
                plan = PreparedStalkParameterRelations(transport, q["relations"])
                _cache[key] = plan
                while len(_cache) > 4:
                    _cache.popitem(last=False)
            _cache.move_to_end(key)
        chosen = q["selected"]; relations = q["relations"]
    certificate = plan.receipt if plan is not None else None
    conditional = plan.condition(chosen) if chosen else None
    if acquisition is not None and conditional is not None and (
            acquisition["certificate_steps"]+conditional["execution_steps"] > state.configuration["limits"][2]):
        raise ValueError("complete selected context family exceeds work admission")
    fit = state.payload["fit"] if certificate is None else certificate["evidence_fit"]
    keys = state.configuration.get("parameter_space")
    components = ("origin", *("free:"+(str(i) if keys is None else keys["keys"][i]) for i in fit["free_columns"]))
    if q["profile"] in _RULE_QUERIES:
        entries = plan.sparse_residual() if plan is not None else ()
        if len(entries) > state.configuration["limits"][1]:
            raise ValueError("sparse context rule coefficients exceed entry admission")
        space = CoordinateSpace("sparse-transport-relation-component",
            tuple(relations[i][0]+":"+components[j] for i, j, _ in entries))
        values = np.asarray(tuple(value for _, _, value in entries), dtype=object)
        return q, certificate, conditional, space, (), values, acquisition
    space = CoordinateSpace("transport-parameter-relation", tuple(row[0] for row in relations))
    axes = (CoordinateSpace("evidence-affine-component", components),)
    if len(space.keys)*len(components) > state.configuration["limits"][1]:
        raise ValueError("complete parameter relation output exceeds entry admission")
    values = np.full((len(space.keys), len(components)), Fraction(0), dtype=object)
    if certificate is not None:
        values[:, 0] = certificate["residual_origin"]
        for row, column, value in certificate["residual_directions"]:
            values[row, column+1] = value
    return q, certificate, conditional, space, axes, values, acquisition


def _infer(state, source, inputs):
    from rexgraph.model_codec import _model_checkpoint
    q, certificate, conditional, space, axes, values, acquisition = _analysis(state, source, inputs)
    receipt = dict(profile=_METHOD, model_checkpoint=_model_checkpoint(state), source_archive=source.archive,
        query_declaration=q, certificate=certificate, conditional_family=conditional,
        interpretation="complete-evidence-parameter-relations-v1")
    if acquisition is not None:
        receipt["acquisition"] = acquisition
    return ModelOutput(values, space, axes, state.source, state.coefficient_digest, "rational", _METHOD,
                       grade=None, variance="coordinate", execution_receipt=receipt)


def validate_stalk_parameter_relations_output(output):
    """Replay all residual directions and any explicitly selected intersection."""
    from rexgraph.model_codec import _checkpoint_model
    from rexgraph_cpp import NativeSource
    r = output.execution_receipt
    fields = {"profile", "model_checkpoint", "source_archive", "query_declaration", "certificate", "conditional_family", "interpretation"}
    if (isinstance(r, Mapping) and isinstance(r.get("query_declaration"), Mapping) and
            r["query_declaration"].get("profile") in _CONTEXT_QUERIES):
        fields.add("acquisition")
    if (not isinstance(r, Mapping) or set(r) != fields or
            r["profile"] != _METHOD or r["interpretation"] != "complete-evidence-parameter-relations-v1"):
        raise ValueError("complete original parameter relation receipt required")
    state = _checkpoint_model(r["model_checkpoint"])
    if (not isinstance(state, ModelState) or state.adapter != "cpp_stalk_transport" or
            output.method != _METHOD or output.arithmetic != "rational" or output.dependencies or
            output.grade is not None or output.variance != "coordinate" or
            not output.source.matches(state.source) or output.model_digest != state.coefficient_digest):
        raise ValueError("parameter relation output differs from its actual source and model")
    maximum = state.configuration["max_bytes"]
    if type(r["source_archive"]) is not bytes or len(r["source_archive"]) > maximum:
        raise ValueError("bounded original native source archive required")
    source = NativeSource(r["source_archive"], max_bytes=maximum)
    _, certificate, conditional, space, axes, values, acquisition = _analysis(state, source, r["query_declaration"])
    if (output.space != space or output.axes != axes or not np.array_equal(output.values, values) or
            not _receipt_matches(certificate, r["certificate"]) or
            not _receipt_matches(conditional, r["conditional_family"]) or
            not _receipt_matches(acquisition, r.get("acquisition"))):
        raise ValueError("parameter relation output differs from complete native observation replay")


def stalk_parameter_relation(output, name):
    """Read one complete residual family and its evidence classification."""
    output.check_state(); validate_stalk_parameter_relations_output(output)
    r = output.execution_receipt
    relations = r["acquisition"]["relations"] if "acquisition" in r else r["query_declaration"]["relations"]
    names = tuple(relation[0] for relation in relations)
    if type(name) is not str or name not in names:
        raise ValueError("select an original declared parameter relation name")
    row = names.index(name)
    if r["query_declaration"]["profile"] in _RULE_QUERIES:
        fit = r["certificate"]["evidence_fit"]
        keys = fit.get("parameter_space")
        components = ("origin", *("free:"+(str(i) if keys is None else keys[1][i]) for i in fit["free_columns"]))
        family = [Fraction(0)]*len(components); family[0] = r["certificate"]["residual_origin"][row]
        for i, j, value in r["certificate"]["residual_directions"]:
            if i == row:
                family[j+1] = value
    else:
        components = output.axes[0].keys; family = output.values[row]
    return freeze_tree(dict(name=name, declaration=relations[row],
        components=components, residual_family=tuple(family),
        status=r["certificate"]["statuses"][row], evidence_scope=r["certificate"]["evidence_scope"]))


def stalk_context_hypotheses(output):
    """Read complete acquired star maps and whole block evidence classifications."""
    output.check_state(); validate_stalk_parameter_relations_output(output)
    if "acquisition" not in output.execution_receipt:
        raise ValueError("acquired context hypothesis output required")
    return output.execution_receipt["acquisition"]


@dataclass(frozen=True, init=False, repr=False, eq=False)
class PreparedStalkContextRules:
    """Own a verified source and native conditional proof for repeated field actions."""
    _transport: object
    _plan: object
    _selected: tuple
    _metadata: object
    _has_proof: bool
    _output_digest: str

    def __init__(self, output):
        from rexgraph.model_codec import _checkpoint_model
        from rexgraph_cpp import NativeSource, PreparedStalkParameterRelations
        if hasattr(self, "_plan"):
            raise RuntimeError("context rule owner is immutable")
        if not isinstance(output, ModelOutput):
            raise TypeError("learned context rule ModelOutput required")
        output.check_state(); validate_stalk_parameter_relations_output(output)
        r = output.execution_receipt
        if r["query_declaration"]["profile"] not in _RULE_QUERIES:
            raise ValueError("learned context rule output required")
        state = _checkpoint_model(r["model_checkpoint"])
        source = NativeSource(r["source_archive"], max_bytes=state.configuration["max_bytes"])
        transport = _prepare(state, source)
        plan, acquisition, chosen = _context_analysis(state, transport, r["query_declaration"])
        has_proof = plan is not None
        if plan is None:
            plan = PreparedStalkParameterRelations(transport, (("unrestricted-family", (), 0),), allow_constants=True)
        scope = ("no-admitted-context-rule" if not chosen else
            "conditional-on-covered-observed-field-invariance" if r["query_declaration"]["profile"] == _OBSERVED_VIEW_RULES_QUERY else
            "conditional-on-covered-context-invariance" if r["query_declaration"]["profile"] in _VIEW_RULE_QUERIES else
            "conditional-on-observed-context-invariance")
        metadata = freeze_tree(dict(model_digest=state.coefficient_digest, source_reference=state.source.as_record(),
            selected_blocks=acquisition["selected_blocks"], transfer_scope=scope))
        for name, value in (("_transport", transport), ("_plan", plan), ("_selected", chosen),
                ("_metadata", metadata), ("_has_proof", has_proof), ("_output_digest", output.coefficient_digest)):
            object.__setattr__(self, name, value)

    @property
    def receipt(self):
        """Return the retained source, model, rule selection and output identities."""
        self._require()
        return freeze_tree(dict(self._metadata, output_digest=self._output_digest, selected=self._selected))

    def _require(self):
        if not hasattr(self, "_plan"):
            raise RuntimeError("context rule owner is not initialized")

    def apply(self, field):
        """Apply the complete conditional family on original upper stalks."""
        self._require()
        response = self._plan.apply(field, self._selected) if self._has_proof else self._transport.apply(field)
        return freeze_tree(dict(response, selected=self._selected, profile="conditional-whole-stalk-affine-response-v1", **self._metadata))

    def restrict_fields(self, observations):
        """Intersect complete observed fields without refitting or replacing evidence."""
        self._require()
        return freeze_tree(dict(self._plan.restrict_fields(observations, self._selected), **self._metadata))


def stalk_context_rule_response(output, field):
    """Apply learned conditional maps while retaining every nonzero field direction."""
    return PreparedStalkContextRules(output).apply(field)


def stalk_context_rule_restriction(output, observations):
    """Check complete meaning fields against the joint learned conditional family."""
    return PreparedStalkContextRules(output).restrict_fields(observations)


def validate_stalk_context_restriction(output, restriction):
    """Replay a complete field restriction against its original model output."""
    if not isinstance(restriction, Mapping) or "observations" not in restriction:
        raise ValueError("complete context field restriction required")
    expected = stalk_context_rule_restriction(output, restriction["observations"])
    if not _receipt_matches(expected, restriction):
        raise ValueError("context field restriction differs from complete native replay")
    return True
