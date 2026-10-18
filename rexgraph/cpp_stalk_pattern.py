"""Compact native inference patterns with external training provenance."""
from rexgraph.cpp_stalk_transport import _prepare
from rexgraph.cpp_stalk_parameter_relations import _query, _VIEW_RULE_QUERIES, _OBSERVED_VIEW_RULES_QUERY, _TEMPLATE_VIEW_RULES_QUERY


def compile_stalk_context_pattern_scope(state, source, query, *, source_namespace):
    """Retain all observed original stars and input images for serving scope checks."""
    query = _query(state, query)
    if query["profile"] not in _VIEW_RULE_QUERIES:
        raise ValueError("covered context view rule query required")
    native = _prepare(state, source)
    return native.observed_pattern_scope(query["frames"], source_namespace=source_namespace,
        evidence_digest=state.coefficient_digest, max_bytes=state.configuration["max_bytes"])


def compile_stalk_context_pattern(state, source, query, origin, *, model_namespace, coordinate_ids):
    """Export an admitted training star on the common input image of its agreeing peers."""
    query = _query(state, query)
    if query["profile"] not in _VIEW_RULE_QUERIES:
        raise ValueError("covered context view rule query required")
    if query["profile"] == _TEMPLATE_VIEW_RULES_QUERY:
        raise ValueError("declared templates require the pattern bank exporter")
    native = _prepare(state, source)
    return native.learned_pattern(query["training"], origin, query["frames"],
        model_namespace=model_namespace, coordinate_ids=coordinate_ids,
        evidence_digest=state.coefficient_digest, minimum_peers=query["minimum_peers"],
        observed_inputs=query["profile"] == _OBSERVED_VIEW_RULES_QUERY)


def compile_stalk_context_pattern_bank(state, source, query, origins, *, model_namespace, coordinate_ids):
    """Export the selected star union and preserve correlations across its observable weights."""
    query = _query(state, query)
    if query["profile"] not in _VIEW_RULE_QUERIES:
        raise ValueError("covered context view rule query required")
    native = _prepare(state, source)
    return native.learned_pattern_bank(query["training"], origins, query["frames"],
        model_namespace=model_namespace, coordinate_ids=coordinate_ids,
        evidence_digest=state.coefficient_digest, minimum_peers=query["minimum_peers"],
        observed_inputs=query["profile"] == _OBSERVED_VIEW_RULES_QUERY,
        declared_templates=query["profile"] == _TEMPLATE_VIEW_RULES_QUERY)


def prepare_stalk_context_pattern_bank(state, source, query):
    """Retain native training admissions and reuse their complete proof for banks."""
    query = _query(state, query)
    if query["profile"] not in _VIEW_RULE_QUERIES or query["targets"]:
        raise ValueError("training context view rule query without target rows required")
    native = _prepare(state, source)
    return native.prepare_pattern_bank(query["training"], query["frames"],
        evidence_digest=state.coefficient_digest, minimum_peers=query["minimum_peers"],
        observed_inputs=query["profile"] == _OBSERVED_VIEW_RULES_QUERY,
        declared_templates=query["profile"] == _TEMPLATE_VIEW_RULES_QUERY)
