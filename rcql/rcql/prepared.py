"""Bounded process-local result owners over the ordinary checked RCDB cache."""
from collections import OrderedDict
from copy import deepcopy
from dataclasses import replace

from .program_codec import ImplementationSnapshot
from .query_cache import QueryCache, _declared_reference, _fingerprint
from .source_context import field_references


def _private_result(result):
    # Keep whole value owners. Copy only finite receipt metadata that a caller
    # can mutate; serializing fields here would discard the prepared owners.
    return replace(result, native_plan=deepcopy(result.native_plan),
                   provenance=deepcopy(result.provenance), execution=deepcopy(result.execution))


class PreparedQueryCache(QueryCache):
    """Retain up to ``max_results`` complete results on this preparation.

    Every use follows QueryCache's ordinary typed planning, source/publication,
    policy, arithmetic and evidence validation. Durable misses and cold recovery
    use the existing RCDB cache. Equal output coefficients never authorize reuse.
    The bound counts results, not their bytes.
    """

    def __init__(self, store, *, namespace="rcql_result", max_results=32):
        super().__init__(store, namespace=namespace)
        if type(max_results) is not int or max_results < 1:
            raise ValueError("max_results must be a positive integer")
        self.max_results = max_results
        self._results = OrderedDict()
        self._implementation = ImplementationSnapshot()

    def _implementation_identity(self):
        return self._implementation.check()

    def _read_result(self, record_id, key, references, cpp_declarations):
        retained = self._results.get(key)
        if retained is None:
            return super()._read_result(record_id, key, references, cpp_declarations)
        original, fingerprint = retained
        if _fingerprint(original.values) != fingerprint:
            del self._results[key]
            raise ValueError("retained result changed after preparation")
        for reference in field_references(original.values):
            if not _declared_reference(reference, references, cpp_declarations):
                raise ValueError("retained result has an unbound source dependency")
        self._results.move_to_end(key)
        result = _private_result(original)
        event = {"node": "cache", "operator": "RESULT_REUSE", "method_status": "observed",
                 "methods": [{"method": "validated-prepared-result-reuse", "cache_key": key}],
                 "exactness": "structural"}
        return replace(result,
            native_plan=dict(result.native_plan or {}, cache={"key": key, "hit": True}),
            provenance=tuple(dict(p, reuse={"key": key, "original_execution": p.get("execution")})
                             for p in result.provenance), execution=(event,))

    def _retain_result(self, key, result):
        # Preserve the original execution receipt, rather than recursively
        # retaining the receipt of each subsequent reuse.
        if key not in self._results:
            self._results[key] = (_private_result(result), _fingerprint(result.values))
        self._results.move_to_end(key)
        while len(self._results) > self.max_results:
            self._results.popitem(last=False)

    def clear(self):
        """Release retained result owners; durable records remain available."""
        with self._lock:
            self._results.clear()
