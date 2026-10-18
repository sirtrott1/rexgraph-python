"""Immutable envelopes for native whole field result bytes."""
from collections.abc import Mapping
from hashlib import sha256

FIELD_ARCHIVE_SCHEMA = "rexgraph.relational-field-program-result.v1"
FIELD_ARCHIVE_METHOD = "exact-relational-field-program-v1"
ACCESSION_FIELD_ARCHIVE_SCHEMA = "rexgraph.relational-accession-field-result.v1"
ACCESSION_FIELD_ARCHIVE_METHOD = "exact-relational-accession-field-program-v1"
AFFINE_FIELD_ARCHIVE_SCHEMA = "rexgraph.relational-affine-field-result.v1"
AFFINE_FIELD_ARCHIVE_METHOD = "exact-relational-affine-field-program-v1"
EVALUATION_FIELD_ARCHIVE_SCHEMA = "rexgraph.relational-affine-evaluation.v1"
EVALUATION_FIELD_ARCHIVE_METHOD = "exact-relational-affine-evaluation-v1"
COMPARISON_FIELD_ARCHIVE_SCHEMA = "rexgraph.relational-comparison-field-result.v1"
COMPARISON_FIELD_ARCHIVE_METHOD = "exact-relational-comparison-field-program-v1"
AVAILABLE_FIELD_ARCHIVE_SCHEMA = "rexgraph.available-relational-field-result.v1"
AVAILABLE_FIELD_ARCHIVE_METHOD = "exact-available-relational-field-program-v1"
AVAILABLE_AFFINE_FIELD_ARCHIVE_SCHEMA = "rexgraph.available-relational-affine-field-result.v1"
AVAILABLE_AFFINE_FIELD_ARCHIVE_METHOD = "exact-available-relational-affine-field-program-v1"
AVAILABLE_EVALUATION_FIELD_ARCHIVE_SCHEMA = "rexgraph.available-relational-affine-evaluation.v1"
AVAILABLE_EVALUATION_FIELD_ARCHIVE_METHOD = "exact-available-relational-affine-evaluation-v1"
LOCAL_FIELD_ARCHIVE_SCHEMA = "rexgraph.available-relational-local-result.v1"
LOCAL_FIELD_ARCHIVE_METHOD = "exact-available-relational-local-solve-v1"
EXTENSION_FIELD_ARCHIVE_SCHEMA = "rexgraph.available-relational-extension-family.v1"
EXTENSION_FIELD_ARCHIVE_METHOD = "exact-available-relational-extension-family-v1"
EXTENSION_EVALUATION_FIELD_ARCHIVE_SCHEMA = "rexgraph.available-relational-extension-evaluation.v1"
EXTENSION_EVALUATION_FIELD_ARCHIVE_METHOD = "exact-available-relational-extension-evaluation-v1"
FINITE_FIELD_ARCHIVE_SCHEMA = "rexgraph.available-relational-finite-field-family.v1"
FINITE_FIELD_ARCHIVE_METHOD = "exact-available-relational-finite-field-family-v1"
FINITE_EVALUATION_FIELD_ARCHIVE_SCHEMA = "rexgraph.available-relational-finite-field-evaluation.v1"
FINITE_EVALUATION_FIELD_ARCHIVE_METHOD = "exact-available-relational-finite-field-evaluation-v1"
LITERAL_FINITE_FIELD_ARCHIVE_SCHEMA = "rexgraph.available-relational-literal-finite-field-family.v1"
LITERAL_FINITE_FIELD_ARCHIVE_METHOD = "exact-available-relational-literal-finite-field-family-v1"
LITERAL_FINITE_EVALUATION_FIELD_ARCHIVE_SCHEMA = "rexgraph.available-relational-literal-finite-field-evaluation.v1"
LITERAL_FINITE_EVALUATION_FIELD_ARCHIVE_METHOD = "exact-available-relational-literal-finite-field-evaluation-v1"
FINITE_FIELD_ARCHIVE_SCHEMAS = {FINITE_FIELD_ARCHIVE_SCHEMA, FINITE_EVALUATION_FIELD_ARCHIVE_SCHEMA,
                              LITERAL_FINITE_FIELD_ARCHIVE_SCHEMA, LITERAL_FINITE_EVALUATION_FIELD_ARCHIVE_SCHEMA}
OPERATOR_FIELD_ARCHIVE_SCHEMA = "rexgraph.available-relational-finite-operator-family.v1"
OPERATOR_FIELD_ARCHIVE_METHOD = "exact-available-relational-finite-operator-family-v1"
OPERATOR_EVALUATION_FIELD_ARCHIVE_SCHEMA = "rexgraph.available-relational-finite-operator-evaluation.v1"
OPERATOR_EVALUATION_FIELD_ARCHIVE_METHOD = "exact-available-relational-finite-operator-evaluation-v1"
OPERATOR_FIELD_ARCHIVE_SCHEMAS = {OPERATOR_FIELD_ARCHIVE_SCHEMA, OPERATOR_EVALUATION_FIELD_ARCHIVE_SCHEMA}
KNOWN_PRODUCT_FIELD_ARCHIVE_SCHEMA = "rexgraph.available-relational-known-product-family.v1"
KNOWN_PRODUCT_FIELD_ARCHIVE_METHOD = "exact-available-relational-known-product-family-v1"
CONDITIONED_FIELD_ARCHIVE_SCHEMA = "rexgraph.conditioned-field-family.v1"
CONDITIONED_FIELD_ARCHIVE_METHOD = "exact-conditioned-field-program-v1"

FIELD_ARCHIVE_METHODS = {LITERAL_FINITE_FIELD_ARCHIVE_SCHEMA: LITERAL_FINITE_FIELD_ARCHIVE_METHOD,
                         LITERAL_FINITE_EVALUATION_FIELD_ARCHIVE_SCHEMA: LITERAL_FINITE_EVALUATION_FIELD_ARCHIVE_METHOD,"rexgraph.conditioned-occurrence-field-family.v1": "exact-conditioned-occurrence-program-v1", "rexgraph.conditioned-joint-field-family.v1": "exact-conditioned-joint-field-family-v1", CONDITIONED_FIELD_ARCHIVE_SCHEMA: CONDITIONED_FIELD_ARCHIVE_METHOD,
                         KNOWN_PRODUCT_FIELD_ARCHIVE_SCHEMA: KNOWN_PRODUCT_FIELD_ARCHIVE_METHOD,FINITE_FIELD_ARCHIVE_SCHEMA: FINITE_FIELD_ARCHIVE_METHOD,
                         OPERATOR_FIELD_ARCHIVE_SCHEMA: OPERATOR_FIELD_ARCHIVE_METHOD,
                         OPERATOR_EVALUATION_FIELD_ARCHIVE_SCHEMA: OPERATOR_EVALUATION_FIELD_ARCHIVE_METHOD,
                         FINITE_EVALUATION_FIELD_ARCHIVE_SCHEMA: FINITE_EVALUATION_FIELD_ARCHIVE_METHOD,
                         EXTENSION_FIELD_ARCHIVE_SCHEMA: EXTENSION_FIELD_ARCHIVE_METHOD,
                         EXTENSION_EVALUATION_FIELD_ARCHIVE_SCHEMA: EXTENSION_EVALUATION_FIELD_ARCHIVE_METHOD,
                         FIELD_ARCHIVE_SCHEMA: FIELD_ARCHIVE_METHOD,
                         ACCESSION_FIELD_ARCHIVE_SCHEMA: ACCESSION_FIELD_ARCHIVE_METHOD,
                         AFFINE_FIELD_ARCHIVE_SCHEMA: AFFINE_FIELD_ARCHIVE_METHOD,
                         EVALUATION_FIELD_ARCHIVE_SCHEMA: EVALUATION_FIELD_ARCHIVE_METHOD,
                         COMPARISON_FIELD_ARCHIVE_SCHEMA: COMPARISON_FIELD_ARCHIVE_METHOD,
                         AVAILABLE_FIELD_ARCHIVE_SCHEMA: AVAILABLE_FIELD_ARCHIVE_METHOD,
                         AVAILABLE_AFFINE_FIELD_ARCHIVE_SCHEMA: AVAILABLE_AFFINE_FIELD_ARCHIVE_METHOD,
                         AVAILABLE_EVALUATION_FIELD_ARCHIVE_SCHEMA: AVAILABLE_EVALUATION_FIELD_ARCHIVE_METHOD,
                         LOCAL_FIELD_ARCHIVE_SCHEMA: LOCAL_FIELD_ARCHIVE_METHOD}


def freeze_field_archive(value):
    """Check the portable envelope without claiming native replay validation."""
    from rexgraph.model_state import freeze_tree

    if not isinstance(value, Mapping):
        raise TypeError("field archive requires an immutable value mapping")
    carrier = ("operator_sha256" if value.get("schema") in OPERATOR_FIELD_ARCHIVE_SCHEMAS else
               "map_sha256" if value.get("schema") in FINITE_FIELD_ARCHIVE_SCHEMAS else "program_sha256")
    if set(value) != {"schema", "data", "sha256", carrier, "engine_receipt"}:
        raise ValueError("field archive envelope has unknown or missing fields")
    if value["schema"] not in FIELD_ARCHIVE_METHODS:
        raise ValueError("unsupported field archive schema")
    if type(value["data"]) is not bytes:
        raise TypeError("field archive requires immutable native bytes")
    if not value["data"] or value["sha256"] != sha256(value["data"]).hexdigest():
        raise ValueError("field archive bytes differ from their declared digest")
    program = value[carrier]
    message = f"field archive requires a canonical {carrier.removesuffix('_sha256')} SHA256"
    if type(program) is not str or len(program) != 64:
        raise ValueError(message)
    try:
        canonical = bytes.fromhex(program).hex()
    except ValueError as exc:
        raise ValueError(message) from exc
    if canonical != program:
        raise ValueError(message)
    if not isinstance(value["engine_receipt"], Mapping) or not value["engine_receipt"]:
        raise TypeError("field archive requires its declared engine receipt")
    return freeze_tree(value)
