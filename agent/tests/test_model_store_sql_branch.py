"""Model archetype inputs use selected RCDB records rather than SQL tables."""
import numpy as np
import pytest
from agent.models.store import load_bundle
from agent import rcdb


def test_model_input_from_native_record_keeps_its_revision():
    s = rcdb.MemoryStore()
    s.put_record("training", {"X": np.array([[1.,2.],[3.,4.]]), "y": np.array([0,1])}, expected_version=0)
    bundle = load_bundle(s, record_id="training", version=1)
    assert bundle.X.shape == (2,2)
    assert bundle.y.tolist() == [0,1]
    assert bundle.meta["source_record"]["version"] == 1


@pytest.mark.parametrize("uri", ["postgresql://host/db", "sqlite:///unused"])
def test_sql_model_sources_are_not_opened(uri):
    with pytest.raises(ValueError, match="RCDB"):
        load_bundle(uri, table="training")
