"""Construct and analyze relational complexes from application data.

The Agent package provides input adapters, analysis pipelines, sessions,
record storage integration and application services.
"""

#: The agent version, kept here rather than read back from installed metadata so a
#: source checkout reports what it is. Sibling dependency floors move with the
#: shared contracts: declared native state, portable results and exact record
#: metadata require this development series. No release is published by a bump.
#: pyproject.toml and the other four packages must agree.
__version__ = "1.4.0"

from .auto import auto_analyze, auto_rex, auto_rex_text, detect_input_type
from .analysis import analyze
from .engine import DecisionEngine
from .pipeline import AnalysisPipeline
from .rcdb_integration import configure as _configure_rcdb
from .session import Session

# The sibling rcdb package does not import the agent, so this is where the agent's
# activity feed, request scoping, metadata privacy and similarity scoring are handed to
# it. Done at import, so a caller reaching for agent.rcdb gets the agent's behaviour
# without having to know this wiring exists.
_configure_rcdb()

__all__ = [
    "auto_rex",
    "auto_rex_text",
    "auto_analyze",
    "analyze",
    "detect_input_type",
    "AnalysisPipeline",
    "Session",
    "DecisionEngine",
    "rcql_runtime",
    "RelationalSession",
    "FamilyVerifier",
    "LearnedWorker",
    "FiniteProgramFactory",
]


def __getattr__(name):
    """Load the integrated query and record interfaces on first use."""
    if name == "rcql_runtime":
        from .rcql_runtime import rcql_runtime
        return rcql_runtime
    if name == "RelationalSession":
        from .relational import RelationalSession
        return RelationalSession
    if name == "FamilyVerifier":
        from .family_verification import FamilyVerifier
        return FamilyVerifier
    if name == "LearnedWorker":
        from .learned_worker import LearnedWorker
        return LearnedWorker
    if name == "FiniteProgramFactory":
        from .program_factory import FiniteProgramFactory
        return FiniteProgramFactory
    raise AttributeError(name)
