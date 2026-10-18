"""Original linguistic assertions with separate token uses and literal byte charts."""
from dataclasses import dataclass
from fractions import Fraction

from rexgraph_cpp import NativeSource
from rexgraph.model_state import freeze_tree

from rexgraph_cpp import literal_token_chart as token_chart


@dataclass(frozen=True)
class StructuralLanguageSource:
    """Owned relation source and retained use, assertion and byte accessions."""
    source: object
    declaration: object
    lower: tuple
    rows: object
    charts: tuple

    def relation(self, document, assertion):
        return self.rows[document][assertion]

    def accession(self):
        """Retain original declarations and both literal and linguistic identities."""
        return dict(profile="original-structural-language-source-v1", declaration=self.declaration,
                    lower=self.lower, rows=self.rows, charts=self.charts)


def structural_language_source(documents, types, roles, *, namespace=981117):
    """Retain supplied typed assertions without inferring grammar from spelling."""
    types = tuple(types); roles = tuple(roles)
    if (not types or len(set(types)) != len(types) or "use-encoding" in types
            or len(set(roles)) != len(roles) or set(roles) & {"encoded-use", "literal-record"}):
        raise ValueError("distinct declared linguistic types and roles required")
    types += ("use-encoding",); roles += ("encoded-use", "literal-record")
    nodes = {}; relations = []; rows = {}; charts = []; key = 0
    def node(identity):
        return nodes.setdefault(identity, len(nodes))
    def relation(kind, participants):
        nonlocal key
        flags = []
        for identity, role, exists, negative, magnitude in participants:
            key += 1
            flags.append(dict(key=key, lower=node(identity), aspects=(roles.index(role),), exists=exists,
                              negative=negative, magnitude=magnitude))
        row = len(relations)
        relations.append(dict(grade=1, key=row+1, type=types.index(kind), occurrences=tuple(flags)))
        return row
    for ordinal, document in enumerate(documents):
        if set(document) != {"id", "raw", "spans", "assertions"} or not isinstance(document["raw"], bytes):
            raise ValueError("language source requires raw bytes and supplied original assertions")
        identity = document["id"]
        if not isinstance(identity, str) or not identity or identity in rows:
            raise ValueError("distinct original document identities required")
        raw = document["raw"]; raw.decode("utf-8")
        spans = tuple(tuple(span) for span in document["spans"])
        previous = 0; words = []
        for span in spans:
            if (len(span) != 2 or any(type(i) is not int for i in span)
                    or not previous <= span[0] < span[1] <= len(raw)):
                raise ValueError("token spans must respect original ordered UTF8 intervals")
            try:
                words.append(raw[slice(*span)].decode("utf-8"))
            except UnicodeDecodeError as error:
                raise ValueError("token spans must respect original UTF8 codepoint boundaries") from error
            previous = span[1]
        words = tuple(words)
        token, literal, mapping, family = token_chart(raw, words, spans,
            namespace=namespace+1000+3*ordinal, retain_family=True)
        if tuple(literal.field(1001).values) != tuple(raw):
            raise ValueError("literal byte source differs from its original document")
        rows[identity] = {}; encoding = []; uses = []
        for index, span in enumerate(spans):
            use = ("use", identity, index); uses.append(node(use))
            encoding.append(relation("use-encoding", (
                (use, "encoded-use", True, False, 1),
                (("literal-utf8", raw[slice(*span)].hex()), "literal-record", True, False, 1))))
        for assertion in document["assertions"]:
            if set(assertion) != {"id", "type", "participants"} or assertion["type"] not in types[:-1]:
                raise ValueError("closed source assertion with a declared linguistic type required")
            name = assertion["id"]
            if not isinstance(name, str) or not name or name in rows[identity]:
                raise ValueError("distinct original assertion identities required")
            participants = []
            for participant in assertion["participants"]:
                if set(participant) != {"token", "concept", "role", "exists", "negative", "magnitude"}:
                    raise ValueError("closed original participant assertion required")
                token_index, concept = participant["token"], participant["concept"]
                if (token_index is None) == (concept is None):
                    raise ValueError("participant must name one token use or declared concept")
                if token_index is not None:
                    if type(token_index) is not int or not 0 <= token_index < len(spans):
                        raise ValueError("source participant outside original token uses")
                    lower = ("use", identity, token_index)
                else:
                    if not isinstance(concept, str) or not concept:
                        raise ValueError("declared concept identity required")
                    lower = ("concept", concept)
                if (participant["role"] not in roles[:-2] or type(participant["exists"]) is not bool
                        or type(participant["negative"]) is not bool or isinstance(participant["magnitude"], bool)
                        or not isinstance(participant["magnitude"], (int, Fraction)) or participant["magnitude"] < 0):
                    raise ValueError("original participant requires a role and exact composite coefficient")
                participants.append((lower, participant["role"], participant["exists"],
                                     participant["negative"], participant["magnitude"]))
            rows[identity][name] = relation(assertion["type"], participants)
        charts.append(dict(document=identity, raw_utf8_hex=raw.hex(), spans=spans, uses=tuple(uses),
            encoding_rows=tuple(encoding), byte_source_archive=literal.archive,
            token_source_archive=token.archive, token_to_byte_family=family,
            mapping="use-to-literal-accession", chain_mapping="complete-token-to-byte-family"))
    declaration = dict(profile="declared-role-source-v1", namespace=namespace, semantic_scale=2,
        max_grade=1, role_aspects=roles,
        types=tuple(dict(name=n, min_arity=0, max_arity=0) for n in types),
        witnesses=tuple(range(len(nodes))), relations=tuple(relations), fields=(), metrics=(), carriers=())
    source = NativeSource.from_declaration(declaration)
    return StructuralLanguageSource(source, freeze_tree(declaration), tuple(nodes), freeze_tree(rows), freeze_tree(charts))
