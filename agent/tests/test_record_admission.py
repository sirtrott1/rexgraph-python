"""All Agent store entry points refuse SQL before reading or writing records."""
from types import SimpleNamespace
from unittest.mock import MagicMock
import pytest


class SQLProvider:
    backend = "sql"
    def __getattr__(self, name):
        raise AssertionError("SQL provider accessed: " + name)


def test_wrapped_sql_provider_is_refused_and_ordinary_mock_does_not_recurse():
    from agent.record_access import require_store
    with pytest.raises(ValueError, match="SQL"):
        require_store(SimpleNamespace(_inner=SQLProvider()))
    selected = MagicMock()
    assert require_store(selected) is selected


@pytest.mark.parametrize("component", ["courier", "hive-schema", "query-manager", "change-source",
                                       "online-loop", "annotations", "molecules", "corpus", "network", "hive"])
def test_supplied_sql_store_cannot_bypass_agent_uri_admission(component):
    from agent.hive import Hive
    store = SQLProvider()
    with pytest.raises(ValueError, match="SQL"):
        if component == "courier":
            from agent.courier import Courier
            Courier("admission").attach_store("source", store)
        elif component == "hive-schema":
            from agent.hive_schema import HiveSchema
            HiveSchema(Hive("admission"), store=store)
        elif component == "query-manager":
            from agent.query_manager import QueryManager
            QueryManager(store=store)
        elif component == "change-source":
            from agent.temporal_loop import ChangeSource
            ChangeSource(store)
        elif component == "online-loop":
            from agent.temporal_loop import OnlineLoop
            OnlineLoop(store)
        elif component == "annotations":
            from agent.adapters.annotation_store import ingest_annotations
            ingest_annotations(store, "never-read", document_id="source")
        elif component == "molecules":
            from agent.adapters.molecular_store import ingest_molecules
            ingest_molecules(store, "never-read", document_id="source")
        elif component == "corpus":
            from agent.corpus import CorpusBuilder
            CorpusBuilder.from_store(store)
        elif component == "network":
            from agent.hive_network import HiveNetwork
            HiveNetwork().persist(store)
        else:
            hive = Hive("admission")
            hive.add_worker("count", lambda data, **kwargs: len(data), capability="analyze", worker_type="count")
            hive.persist(store)


def test_unregistered_store_scheme_never_reaches_the_sql_fallback(monkeypatch):
    from agent import rcdb
    import rcdb.core as core
    def forbidden(*args, **kwargs):
        pytest.fail("Agent reached the generic SQL fallback")
    monkeypatch.setattr(core, "SQLStore", forbidden)
    for uri in ("object://unregistered", "unknown-driver://host/db"):
        with pytest.raises(ValueError):
            rcdb.open_store(uri)
