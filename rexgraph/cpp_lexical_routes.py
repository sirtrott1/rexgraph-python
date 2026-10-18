"""Exact original dictionary paths and their independent learned field responses."""
from hashlib import sha256
from collections import OrderedDict
from threading import RLock
from types import MappingProxyType
from dataclasses import dataclass

from rexgraph.model_state import freeze_tree, thaw_tree


@dataclass(frozen=True)
class LexicalContext:
    """Owned original context, dictionary alternatives and closed encoding maps."""
    corpus: object
    lookup: object
    bindings: tuple
    document: str

    def as_record(self):
        """Return portable source bytes, accessions and complete map witnesses."""
        maps = []; indices = {}; bindings = []
        for binding in self.bindings:
            key = binding["first"], binding["last"], binding["spelling"]
            index = None
            if binding["map"] is not None:
                if key not in indices:
                    indices[key] = len(maps); maps.append(binding["map"])
                index = indices[key]
            bindings.append(dict((k,v) for k,v in binding.items() if k != "map") | dict(map_index=index))
        return thaw_tree(dict(profile="owned-lexical-context-v1", document=self.document,
            namespace=self.corpus.declaration["namespace"], source_archive=self.corpus.source.archive,
            accession=self.corpus.accession(), lookup=self.lookup, bindings=tuple(bindings), maps=tuple(maps)))


class LexicalRouteCatalog:
    """Verify literal charts and own all retained lemma and form sense paths."""
    def __init__(self, original, *, limits=(1000000, 64000000, 16384),
                 max_bytes=256*1024*1024):
        from rexgraph_cpp import PreparedRelationGuards, PreparedLexicalRoutes
        from rexgraph.linguistic_accession import validate_linguistic_accession
        corpus = original.corpus
        owner = PreparedRelationGuards(corpus.source.archive, limits=limits, max_bytes=max_bytes)
        validate_linguistic_accession(corpus.accession(), corpus.source, owner=owner, max_bytes=max_bytes)
        charts = {chart["document"]: chart for chart in corpus.charts}
        spellings = []
        spelling_charts = []
        for document in original.text_owners:
            row = corpus.relation(document, "text-owner")
            declaration = corpus.declaration["relations"][row]
            kind = corpus.declaration["types"][declaration["type"]]["name"]
            if kind not in {"record-lemma", "record-form"}: continue
            chart = charts[document]
            spellings.append(dict(row=row, raw=bytes.fromhex(chart["raw_utf8_hex"]),
                spans=chart["spans"], uses=chart["uses"], encoding_rows=chart["encoding_rows"]))
            spelling_charts.append(freeze_tree(dict(spans=chart["spans"], encoding_rows=chart["encoding_rows"])))
        self._native = PreparedLexicalRoutes(owner, spellings)
        self._original = original
        self._owner = owner
        self._spellings = tuple(spelling_charts)
        self._namespace = corpus.source.identity["namespace"]
        self._limits = tuple(limits)
        self._max_bytes = max_bytes
        self._routes = freeze_tree(self._native.routes)
        self._source_digest = sha256(corpus.source.archive).hexdigest()
        self._source_reference = freeze_tree(corpus.source.identity)
        self._banks = OrderedDict()
        self._lock = RLock()

    @property
    def receipt(self):
        return freeze_tree(self._native.receipt)

    @property
    def routes(self):
        return self._routes

    def context_fields(self, *, lexical_kinds=False, phrases=True, frame_axes=None):
        """Prepare exact named token and phrase fields over this dictionary owner."""
        from rexgraph_cpp import PreparedLexicalContextFields
        return PreparedLexicalContextFields(self._native,self._source_digest,
            lexical_kinds=lexical_kinds,phrases=phrases,frame_axes=frame_axes)

    def lookup(self, raw, spans=None, *, candidate_limit=4096, work_limit=None):
        """Keep every original path matching complete UTF8 interval boundaries.

        Omitted intervals use maximal ASCII whitespace cuts. Matches preserve
        case, punctuation, internal whitespace and overlapping phrases. The
        result names the retained dictionary scope and leaves meaning unresolved.
        """
        from rexgraph_cpp import literal_whitespace_spans
        if not isinstance(raw, bytes): raise TypeError("original query bytes required")
        if not 1 <= len(raw) <= 8192: raise ValueError("bounded original query bytes required")
        raw.decode("utf-8")
        spans = literal_whitespace_spans(raw) if spans is None else spans
        checked = []
        for span in spans:
            if len(checked) >= 128 or len(span) != 2: raise ValueError("bounded original query intervals required")
            start, end = span
            if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(raw):
                raise ValueError("original UTF8 query interval required")
            raw[start:end].decode("utf-8")
            checked.append((start, end))
        spans = tuple(checked)
        report = self._native.lookup(raw, spans, candidate_limit=candidate_limit, work_limit=work_limit)
        metadata = freeze_tree(dict(report, raw=raw, spans=spans, coverage="retained-original-spellings",
            meaning_status="unresolved", unmatched=tuple(i for i in range(len(spans))
                if not any(use["first"] <= i <= use["last"] for use in report["uses"]))))
        return MappingProxyType(dict(metadata, source_reference=self._source_reference))

    def context(self, raw, spans=None, *, document="query", namespace=981250,
                candidate_limit=4096, work_limit=None):
        """Retain one ordered original context and all exact dictionary accessions.

        Each token use is distinct. One context column retains the document
        owner and every ordered use. Encoding maps cover each matching closed
        carrier separately. Different token partitions remain explicit gaps.
        No semantic map is inferred from a byte match.
        """
        from rexgraph.linguistic_source import structural_language_source
        from rexgraph_cpp import PreparedRelationGuards
        if (not isinstance(document, str) or not document or len(document.encode()) > 200
                or type(namespace) is not int or not 1 <= namespace < (1 << 64)-1002):
            raise ValueError("bounded original context identity and namespace required")
        if namespace == self._namespace:
            raise ValueError("context and dictionary require separate source namespaces")
        report = self.lookup(raw, spans, candidate_limit=candidate_limit, work_limit=work_limit)
        spans = report["spans"]
        roles = ("context-owner", *("position:"+str(i) for i in range(len(spans))))
        participants = (dict(token=None, concept=document, role="context-owner", exists=True,
            negative=True, magnitude=1), *(dict(token=i, concept=None, role=roles[i+1], exists=True,
                negative=False, magnitude=1) for i in range(len(spans))))
        corpus = structural_language_source((dict(id=document, raw=raw, spans=spans,
            assertions=(dict(id="context", type="ordered-context", participants=participants),)),),
            ("ordered-context",), roles, namespace=namespace)
        owner = PreparedRelationGuards(corpus.source.archive, limits=self._limits, max_bytes=self._max_bytes)
        chart = corpus.charts[0]
        cache = {}; bindings = []
        remaining = self._limits[1] if work_limit is None else work_limit
        remaining -= report["execution_steps"]
        for use in report["uses"]:
            for route_index in use["routes"]:
                route = self._routes[route_index]; spelling = route["spelling"]
                key = use["first"], use["last"], spelling
                if key not in cache:
                    target = self._spellings[spelling]; first, last = use["first"], use["last"]
                    offset = use["span"][0]
                    cuts = tuple((a-offset, b-offset) for a, b in spans[first:last+1])
                    if cuts != target["spans"]:
                        cache[key] = freeze_tree(dict(status="partition-mismatch", map=None))
                    else:
                        witness = owner.encoding_accession(self._owner, chart["encoding_rows"][first:last+1],
                            target["encoding_rows"], work_limit=remaining)
                        remaining -= witness["execution_steps"]
                        if not witness["matched"]:
                            raise ValueError("original encoding accession failed: "+witness["reason"])
                        cache[key] = freeze_tree(dict(status="mapped", map=witness))
                metadata = freeze_tree(dict(span=use["span"], first=use["first"], last=use["last"],
                    route=route_index, spelling=spelling))
                bindings.append(MappingProxyType(dict(metadata, **cache[key])))
        return LexicalContext(corpus, report, tuple(bindings), document)

    def read_context(self, record):
        """Replay a saved original context against this exact dictionary owner."""
        if (set(record) != {"profile", "document", "namespace", "source_archive", "accession", "lookup", "bindings", "maps"}
                or record["profile"] != "owned-lexical-context-v1"):
            raise ValueError("closed owned lexical context record required")
        lookup = record["lookup"]
        context = self.context(lookup["raw"], lookup["spans"], document=record["document"], namespace=record["namespace"],
            candidate_limit=max(1, sum(len(u["routes"]) for u in lookup["uses"])))
        if context.as_record() != freeze_tree(record):
            raise ValueError("context differs from its original source and dictionary maps")
        return context

    def responses(self, report, state, source, inputs):
        """Run supplied exact fields for each route's original learned text stars.

        Fields use each pattern's declared complete common carrier. Missing
        patterns or missing inputs remain explicit alternatives. Shared synsets
        may reuse a response without identifying distinct original sense paths.
        This operation does not choose a sense from raw text.
        """
        from rexgraph.cpp_stalk_pattern_model import inspect_stalk_pattern_bank_model, stalk_pattern_bank_query
        from rexgraph_cpp import PreparedRelationGuards
        from rexgraph.model_runtime import infer_model
        replay = self.lookup(report["raw"], report["spans"],
            candidate_limit=max(1, sum(len(u["routes"]) for u in report["uses"])))
        if replay != report: raise ValueError("query differs from its original lexical routes")
        report = replay
        inspected = inspect_stalk_pattern_bank_model(state, source)
        if inspected["source_sha256"] != self._source_digest:
            raise ValueError("learned pattern bank differs from the original dictionary source")
        if any(type(index) is not int or not 0 <= index < len(inspected["pattern_keys"]) for index in inputs):
            raise ValueError("input outside original learned bank patterns")
        key = source.identity["archive_digest"]
        with self._lock:
            patterns = self._banks.get(key)
            if patterns is None:
                owner = PreparedRelationGuards(source.archive, limits=self.receipt["limits"])
                patterns = {}
                for index, cell in self._native.bind_patterns(owner):
                    patterns.setdefault(cell, []).append(index)
                patterns = freeze_tree(patterns); self._banks[key] = patterns
                while len(self._banks) > 4: self._banks.popitem(last=False)
            self._banks.move_to_end(key)
        outputs = {}; alternatives = []
        for use in report["uses"]:
            choices = []
            for route_index in use["routes"]:
                route = self._routes[route_index]; available = tuple(patterns.get(route["synset"], ()))
                acquired = tuple(i for i in available if i in inputs)
                for index in acquired:
                    if index not in outputs:
                        outputs[index] = infer_model(state, source, stalk_pattern_bank_query(state, {index: inputs[index]}))
                choices.append(dict(route=route_index, patterns=available, response_patterns=acquired,
                    status="fields" if acquired else "missing-input" if available else "missing-pattern"))
            alternatives.append(dict(span=use["span"], first=use["first"], last=use["last"], choices=tuple(choices)))
        metadata = freeze_tree(dict(profile="original-lexical-route-field-responses-v1",
            model_digest=state.coefficient_digest, alternatives=tuple(alternatives), meaning_status="unresolved"))
        return MappingProxyType(dict(metadata, query=report, outputs=tuple(sorted(outputs.items()))))
