"""Original linguistic source columns and exact literal byte accessions."""
from collections.abc import Mapping
from fractions import Fraction


def validate_linguistic_accession(accession, source, *, max_bytes=256*1024*1024, owner=None):
    """Replay original linguistic columns, UTF8 fields and complete token maps."""
    from rexgraph_cpp import NativeSource
    if accession is None:
        return
    if (not isinstance(accession, Mapping) or set(accession) != {"profile", "declaration", "lower", "rows", "charts"}
            or accession["profile"] != "original-structural-language-source-v1"):
        raise ValueError("closed original linguistic source accession required")
    if NativeSource.from_declaration(accession["declaration"], max_bytes=max_bytes).archive != source.archive:
        raise ValueError("linguistic accession differs from its complete native source")
    if owner is None:
        from rexgraph_cpp import PreparedRelationGuards
        owner = PreparedRelationGuards(source.archive, max_bytes=max_bytes)
    lower = accession["lower"]
    if len(lower) != len(source.identity["coordinates"][0]) or len(set(lower)) != len(lower):
        raise ValueError("linguistic accession requires distinct original lower identities")
    covered = []; documents = set()
    for chart in accession["charts"]:
        if set(chart) != {"document", "raw_utf8_hex", "spans", "uses", "encoding_rows", "byte_source_archive",
                          "token_source_archive", "token_to_byte_family", "mapping", "chain_mapping"}:
            raise ValueError("closed literal byte and token accession required")
        if chart["mapping"] != "use-to-literal-accession" or chart["chain_mapping"] != "complete-token-to-byte-family":
            raise ValueError("literal accession and complete chain mapping must remain distinct")
        raw = bytes.fromhex(chart["raw_utf8_hex"]); raw.decode("utf-8")
        spans = chart["spans"]; previous = 0
        for span in spans:
            if (len(span) != 2 or any(type(i) is not int for i in span)
                    or not previous <= span[0] < span[1] <= len(raw)):
                raise ValueError("literal token cuts differ from original ordered UTF8 intervals")
            try:
                raw[slice(*span)].decode("utf-8")
            except UnicodeDecodeError as error:
                raise ValueError("literal token cuts differ from original ordered UTF8 intervals") from error
            previous = span[1]
        if not spans or len(chart["uses"]) != len(spans) or len(chart["encoding_rows"]) != len(spans):
            raise ValueError("literal accession must cover each original token use")
        literal = NativeSource(chart["byte_source_archive"], max_bytes=max_bytes)
        token = NativeSource(chart["token_source_archive"], max_bytes=max_bytes)
        if tuple(literal.field(1001).values) != tuple(raw):
            raise ValueError("literal byte field differs from its original UTF8 data")
        n = len(spans); points = tuple(i for span in spans for i in span)
        intervals = (*((2*i, 2*i+1) for i in range(n)), *((2*i+1, 2*i+2) for i in range(n-1)),
                     *((2*i, 2*i+3) for i in range(n-1)))
        family = (tuple(((point, Fraction(1)),) for point in points),
                  tuple(tuple((i, Fraction(1)) for i in range(points[a], points[b])) for a, b in intervals),
                  tuple(() for _ in range(n-1)))
        if chart["token_to_byte_family"] != family:
            raise ValueError("token accession differs from its complete ordered byte interval family")
        token.chain_map(literal, family)
        document = chart["document"]
        if not isinstance(document, str) or not document or document in documents:
            raise ValueError("distinct original literal chart identities required")
        documents.add(document)
        for index, (use, row, span) in enumerate(zip(chart["uses"], chart["encoding_rows"], spans, strict=True)):
            boundary = owner.boundary(row); flags = boundary["occurrences"]
            if (type(use) is not int or not 0 <= use < len(lower) or lower[use] != ("use", document, index)
                    or boundary["type"] != "use-encoding" or len(flags) != 2
                    or flags[0]["lower"] != use or lower[flags[1]["lower"]] != ("literal-utf8", raw[slice(*span)].hex())
                    or flags[0]["roles"] != ("encoded-use",) or flags[1]["roles"] != ("literal-record",)
                    or any(not f["exists"] or f["negative"] or f["coefficient"] != "1" for f in flags)):
                raise ValueError("literal use accession differs from its original encoded boundary")
            covered.append(row)
        if document not in accession["rows"]:
            raise ValueError("literal chart lost its original linguistic assertions")
        covered.extend(accession["rows"][document].values())
    if documents != set(accession["rows"]):
        raise ValueError("each original assertion document requires its byte chart")
    if sorted(covered) != list(range(len(source.identity["coordinates"][1]))):
        raise ValueError("linguistic accession must cover every original relation once")
