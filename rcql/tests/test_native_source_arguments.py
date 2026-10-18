"""Native source operators retain named argument order without execution imports."""
import pytest

from rcql import call, param, parse, query, source
from rcql.native_source_contracts import ARGUMENTS


@pytest.mark.parametrize("name", sorted(ARGUMENTS))
def test_named_native_source_arguments_match_positional_calls(name):
    labels, _ = ARGUMENTS[name]
    text = ",".join(f"{label}=${label}" for label in reversed(labels))
    parsed = parse(f"FROM $r RETURN {name}({text})")
    expected = query(source("r"), call(name, *(param(label) for label in labels)))
    assert parsed == expected
