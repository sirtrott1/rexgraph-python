"""Check canonical literal archives, original interval identities and complete maps."""
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction as Q

import pytest

from rexgraph_cpp import NativeSource, literal_token_chart, literal_whitespace_chart, literal_whitespace_spans
def _relation(grade, key, lowers, signs, start):
    return dict(grade=grade, key=key, type=0, occurrences=tuple(
        dict(key=start+j, lower=value, exists=True, negative=negative, magnitude=1)
        for j, (value, negative) in enumerate(zip(lowers, signs, strict=True))))


def _declaration(namespace, scale, sizes, witnesses, relations, fields=(), extra=()):
    return dict(profile="declared-exact-source-v1", namespace=namespace, semantic_scale=scale,
        max_grade=2, types=(dict(name="ordered-interval", min_arity=2, max_arity=4),),
        witnesses=witnesses, relations=relations, fields=fields, metrics=(),
        carriers=tuple(dict(key=10+k, kind="source_grade", grade=k, coordinates=0, size=n)
                       for k, n in enumerate(sizes))+tuple(extra))


def token_chart(raw, words, spans, *, namespace=920060, retain_family=False):
    return literal_token_chart(raw, words, spans, namespace=namespace, retain_family=retain_family)


def original_chart(raw, words, spans, namespace=920060):
    """Construct the original interval declarations and complete sparse grade maps."""
    n, count = len(raw), len(words)
    octets = NativeSource.from_declaration(_declaration(namespace, 0, (n+1, n, 0), tuple(range(n+1)),
        tuple(_relation(1, 20000+i, (i, i+1), (True, False), 2*i) for i in range(n)),
        (dict(key=1001, carrier=11, entries=tuple(enumerate(raw)), support=None),)))
    cuts = tuple(x for span in spans for x in span)
    intervals = [(2*i, 2*i+1) for i in range(count)]
    intervals += [(2*i+1, 2*i+2) for i in range(count-1)]
    intervals += [(2*i, 2*i+3) for i in range(count-1)]
    rows = [_relation(1, 30000+i, (a, b), (True, False), 20000+2*i) for i, (a, b) in enumerate(intervals)]
    rows += [_relation(2, 40000+i, (30000+i, 30000+count+i, 30000+i+1, 30000+2*count-1+i),
                       (False, False, False, True), 40000+4*i) for i in range(count-1)]
    tokens = NativeSource.from_declaration(_declaration(namespace+1, 1, (2*count, 3*count-2, count-1),
        tuple(range(2*count)), rows, extra=(dict(key=20, kind="explicit", grade=0, coordinates=9200601, size=count),)))
    family = (tuple(((cut, Q(1)),) for cut in cuts),
              tuple(tuple((j, Q(1)) for j in range(cuts[a], cuts[b])) for a, b in intervals),
              tuple(() for _ in range(count-1)))
    return tokens, octets, tokens.chain_map(octets, family), family


def document(words, separator=" ", prefix=b"", suffix=b""):
    raw = bytearray(prefix); spans = []
    for i, word in enumerate(words):
        if i: raw.extend(separator.encode())
        start = len(raw); raw.extend(word.encode()); spans.append((start, len(raw)))
    raw.extend(suffix)
    return bytes(raw), words, tuple(spans)


CASES = (
    document(("a",)), document(("a", "a", "a")), document(("猫", "追", "狗"), separator=""),
    document(("Η", "μηχανικός", "όργανο")), document(("الطفل", "قرأ", "الكتاب")),
    document(("a\x00b", "é"), prefix=b"\xff\x00", suffix=b"\xfe"),
    document(tuple("x" for _ in range(128)), separator=""),
)


@pytest.mark.parametrize("case", CASES)
def test_native_matches_original_complete_archives_and_maps(case):
    raw, words, spans = case
    tokens, octets, mapping, family = literal_token_chart(raw, words, spans, retain_family=True)
    original, original_bytes, original_map, original_family = original_chart(*case)
    assert tokens.archive == original.archive and octets.archive == original_bytes.archive
    assert tokens.identity == original.identity and octets.identity == original_bytes.identity
    assert family == original_family
    assert tuple(octets.field(1001).values) == tuple(raw)
    assert mapping.source is tokens and mapping.destination is octets and mapping.denominator == 0
    for grade, keys in enumerate(tokens.identity["coordinates"]):
        values = tuple(Q(i+1, 7) for i in range(len(keys)))
        result = mapping.push(tokens, grade, values)
        assert result == original_map.push(original, grade, values)
        expected = [Q(0) for _ in octets.identity["coordinates"][grade]]
        for value, image in zip(values, family[grade], strict=True):
            for target, coefficient in image: expected[target] += value*coefficient
        assert tuple(expected) == result
    loaded_tokens = NativeSource(tokens.archive); loaded_bytes = NativeSource(octets.archive)
    replay = loaded_tokens.chain_map(loaded_bytes, family)
    assert replay.push(loaded_tokens, 1, (Q(2, 3),)*len(family[1])) == mapping.push(tokens, 1, (Q(2, 3),)*len(family[1]))
    with pytest.raises(ValueError): mapping.push(loaded_tokens, 0, (0,)*len(family[0]))


def test_adjacent_images_keep_distinct_vertices_and_supported_zero_bytes():
    tokens, octets, mapping, family = literal_token_chart(b"a\x00", ("a", "\x00"), ((0, 1), (1, 2)), retain_family=True)
    assert len(tokens.identity["coordinates"][0]) == 4
    assert family[0][1] == family[0][2] == ((1, Q(1)),)
    assert family[1][2] == () and family[2] == ((),)
    assert octets.field(1001).values == (Q(97), Q(0))
    assert mapping.push(tokens, 0, (0, 1, -1, 0)) == (Q(0), Q(0), Q(0))


@pytest.mark.parametrize("words,spans", (
    (("a",), ((0, 0),)), (("a",), ((1, 0),)), (("a",), ((0, 3),)),
    (("a",), ((-1, 0),)), (("a",), ((False, 1),)), (("a",), ((0.0, 1),)),
    (("a",), ((0,),)), (("b",), ((0, 1),)), (("",), ((0, 1),)),
    (("a", "a"), ((0, 1), (0, 1))), (("b", "a"), ((1, 2), (0, 1))),
    (("a",), ()), ((), ()), ((b"a",), ((0, 1),)),
))
def test_invalid_spans_or_words_reject(words, spans):
    with pytest.raises((ValueError, TypeError, OverflowError)):
        literal_token_chart(b"ab", words, spans)


@pytest.mark.parametrize("options", (
    dict(namespace=0), dict(namespace=True), dict(namespace=-1), dict(namespace=1 << 64),
    dict(namespace=(1 << 64)-1), dict(max_bytes=0), dict(max_bytes=1),
    dict(limits=(0, 1, 1)), dict(limits=(1, 1, 1)), dict(limits=(1000000, 1, 16384)),
    dict(limits=(1, 1)), dict(limits=(True, 1, 1)), dict(retain_family=1),
))
def test_namespace_and_resource_admission(options):
    with pytest.raises((ValueError, TypeError, OverflowError, RuntimeError)):
        literal_token_chart(b"a", ("a",), ((0, 1),), **options)


@pytest.mark.parametrize("raw,words,spans", (
    (b"", ("a",), ((0, 1),)), (b"a"*8193, ("a",), ((0, 1),)),
    (b"a"*129, ("a",)*129, tuple((i, i+1) for i in range(129))),
))
def test_source_dimensions_reject(raw, words, spans):
    with pytest.raises(ValueError): literal_token_chart(raw, words, spans)


def test_high_namespace_and_adapter_preserve_original_profile():
    raw, words, spans = document(("猫", "a"))
    native = literal_token_chart(raw, words, spans, namespace=(1 << 64)-2)
    assert native[1].identity["namespace"] == (1 << 64)-2
    assert native[0].identity["namespace"] == (1 << 64)-1
    wrapped = token_chart(raw, words, spans)
    ordinary = literal_token_chart(raw, words, spans)
    assert tuple(s.archive for s in wrapped[:2]) == tuple(s.archive for s in ordinary[:2])


def test_parallel_construction_keeps_owner_binding():
    case = document(("word", "word"))
    with ThreadPoolExecutor(max_workers=4) as workers:
        results = tuple(workers.map(lambda _: literal_token_chart(*case), range(16)))
    assert len({result[0].archive for result in results}) == 1
    for tokens, _, mapping in results:
        assert mapping.source is tokens
        assert mapping.push(tokens, 1, (1, 1, 1, 1))
    with pytest.raises(ValueError): results[0][2].push(results[1][0], 1, (1, 1, 1, 1))


def test_native_path_does_not_normalise_python_byte_declarations(monkeypatch):
    from rexgraph_cpp import _declaration
    def forbidden(*args, **kwargs):
        raise AssertionError("native chart called Python declaration normalization")
    monkeypatch.setattr(_declaration, "normalise", forbidden)
    tokens, octets, _ = literal_token_chart(b"a"*8192, ("a",), ((8191, 8192),))
    assert len(tokens.identity["coordinates"][1]) == 1
    assert tuple(octets.field(1001).values) == (Q(97),)*8192


def test_cold_native_import_and_archive_replay(tmp_path):
    import subprocess
    import sys
    tokens, octets, _, family = literal_token_chart("猫 dog".encode(), ("猫", "dog"), ((0, 3), (4, 7)), retain_family=True)
    (tmp_path/"tokens.rfcxs").write_bytes(tokens.archive)
    (tmp_path/"bytes.rfcxs").write_bytes(octets.archive)
    code = '''
from pathlib import Path
from rexgraph_cpp import NativeSource,literal_token_chart
import sys
assert "rexgraph" not in sys.modules
t,b,m,f=literal_token_chart("猫 dog".encode(),("猫","dog"),((0,3),(4,7)),retain_family=True)
assert t.archive==Path("tokens.rfcxs").read_bytes()
assert b.archive==Path("bytes.rfcxs").read_bytes()
rt,rb=NativeSource(t.archive),NativeSource(b.archive)
assert rt.chain_map(rb,f).push(rt,2,(7,))==()
assert m.push(t,0,(1,-1,1,-1))==(1,0,0,-1,1,0,0,-1)
'''
    child = subprocess.run([sys.executable, "-I", "-c", code], cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert child.returncode == 0, child.stderr


@pytest.mark.parametrize("raw,expected", (
    (b" a\tb\nc\rd\ve\ff ", ((1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12))),
    ("猫 dog".encode(), ((0, 3), (4, 7))),
    ("a\u00a0b".encode(), ((0, 4),)),
    (b"word,word", ((0, 9),)), (b"a\x00b", ((0, 3),)),
    (b"\xff \x80", ((0, 1), (2, 3))),
))
def test_whitespace_policy_preserves_exact_byte_intervals(raw, expected):
    assert literal_whitespace_spans(raw) == expected
    tokens, octets, mapping, spans, family = literal_whitespace_chart(raw, retain_family=True)
    assert spans == expected and tuple(octets.field(1001).values) == tuple(raw)
    assert len(tokens.identity["coordinates"][0]) == 2*len(spans)
    assert family[0] == tuple(((cut, Q(1)),) for span in spans for cut in span)
    replay = tokens.chain_map(octets, family)
    assert replay.push(tokens, 1, (1,)*len(family[1])) == mapping.push(tokens, 1, (1,)*len(family[1]))
    if raw != b"\xff \x80":
        words = tuple(raw[start:end].decode() for start, end in spans)
        original, original_bytes, _, original_family = original_chart(raw, words, spans)
        assert tokens.archive == original.archive and octets.archive == original_bytes.archive
        assert family == original_family


@pytest.mark.parametrize("raw", (b"", b" \t\r\n\v\f", b"a "*129, b"a"*8193))
def test_whitespace_empty_or_oversized_profile_rejects(raw):
    with pytest.raises(ValueError): literal_whitespace_chart(raw)
    with pytest.raises(ValueError): literal_whitespace_spans(raw)


def test_interval_scan_does_not_construct_sources(monkeypatch):
    from rexgraph_cpp import _declaration
    def forbidden(*args, **kwargs):
        raise AssertionError("interval scanning must not construct sources")
    monkeypatch.setattr(_declaration, "normalise", forbidden)
    raw = b" \x00\xff\tword,word\r\n"
    assert literal_whitespace_spans(raw) == ((1, 3), (4, 13))
    assert literal_whitespace_spans(b"x "*128) == tuple((2*i, 2*i+1) for i in range(128))
    assert literal_whitespace_spans(b"x"*8192) == ((0, 8192),)
    with pytest.raises(TypeError): literal_whitespace_spans("x")


def test_whitespace_profile_reuses_namespace_and_resource_admission():
    with pytest.raises(TypeError): literal_whitespace_chart(b"a", retain_family=1)
    with pytest.raises(ValueError): literal_whitespace_chart(b"a", namespace=0)
    with pytest.raises(RuntimeError): literal_whitespace_chart(b"a", limits=(1000000, 1, 16384))
    with pytest.raises(RuntimeError): literal_whitespace_chart(b"a", max_bytes=1)
    assert len(literal_whitespace_chart(b"a")) == 4
    assert len(literal_whitespace_chart(b"a", retain_family=True)) == 5
