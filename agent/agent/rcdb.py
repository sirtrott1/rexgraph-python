"""
agent.rcdb: compatibility surface for the sibling rcdb package.

The implementation moved out of the agent so a store can be installed and reasoned about
without the application. Everything public is re exported here, so the thirty odd modules
and sixty odd test files that import `agent.rcdb` keep working unchanged.

Re exported dynamically rather than by name because the surface is large and a hand
written list is a second place to forget something: a name added to the package and not
to the list would simply stop existing here.
"""
from rcdb import core as _source

for _name in dir(_source):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_source, _name)

del _name, _source

# Export new backends alongside older private core helpers.
import rcdb as _package
for _name in _package.__all__:
    globals()[_name] = getattr(_package, _name)
del _name, _package

import rcdb as _rcdb_package


def register_backend(scheme, factory):
    """Register an explicit RCDB provider for trusted in-process callers."""
    from .record_access import register_store_scheme
    register_store_scheme(scheme)
    return _rcdb_package.register_backend(scheme, factory)


def open_store(uri="memory://", **options):
    """Open an Agent RCDB file or memory provider."""
    from .record_access import check_store_uri, require_store
    return require_store(_rcdb_package.open_store(check_store_uri(uri), **options))


def default_store():
    """Use RCDB's existing scoped default with Agent's provider admission."""
    from .record_access import check_store_uri, require_store
    check_store_uri(_rcdb_package.default_store_uri())
    return require_store(_rcdb_package.default_store())


def SQLStore(*args, **kwargs):
    """Reject the removed SQL-backed Agent storage interface."""
    raise ValueError("Agent uses RCDB file providers; use LocalStore or local://")
