"""Finite program declarations with no executable object decoding."""
import base64
import hashlib
import json
import math
from dataclasses import fields
from fractions import Fraction

from . import ast

_NODES = {cls.__name__: cls for cls in (
    ast.Literal, ast.Parameter, ast.Reference, ast.LetBinding, ast.Call, ast.ListExpr,
    ast.Comparison, ast.MatchBinding, ast.Member, ast.Alias, ast.Query)}


def encode(value, depth=0):
    if depth > 96:
        raise ValueError("program declaration nesting is too deep")
    if type(value).__name__ in _NODES and type(value) is _NODES[type(value).__name__]:
        return {"node": type(value).__name__, "fields":
                {f.name: encode(getattr(value, f.name), depth+1) for f in fields(value)}}
    if value is None or type(value) in (bool, str):
        return {"literal": type(value).__name__, "value": value}
    if type(value) is int:
        return {"literal": "int", "value": hex(value)}
    if isinstance(value, Fraction):
        return {"literal": "rational", "value": [hex(value.numerator), hex(value.denominator)]}
    if type(value) is float and math.isfinite(value):
        return {"literal": "float", "value": value.hex()}
    if type(value) is bytes:
        return {"literal": "bytes", "value": base64.b64encode(value).decode("ascii")}
    from .c_workflow import CValue
    if type(value) is CValue:
        return {"literal":"c-value-declaration","value":[base64.b64encode(value.workflow.data).decode("ascii"),value.index]}
    from .declaration_codec import declaration_types
    if type(value) in declaration_types().values():
        return {"literal": "native-declaration", "value": [type(value).__name__, base64.b64encode(value.data).decode("ascii")]}
    if type(value) in (tuple, list):
        return {"literal": type(value).__name__, "value": [encode(v, depth+1) for v in value]}
    if type(value) is dict and all(type(k) is str for k in value):
        return {"literal": "dict", "value": [[k, encode(value[k], depth+1)] for k in sorted(value)]}
    raise TypeError("program literals must be finite data; bind native values as parameters")


def decode(record, depth=0):
    if depth > 96 or not isinstance(record, dict):
        raise ValueError("invalid program declaration")
    if "node" in record:
        if set(record) != {"node", "fields"} or record["node"] not in _NODES:
            raise ValueError("unknown program syntax node")
        cls = _NODES[record["node"]]
        payload = record["fields"]
        if not isinstance(payload, dict) or set(payload) != {f.name for f in fields(cls)}:
            raise ValueError("program syntax fields do not match the schema")
        return cls(**{k: decode(v, depth+1) for k, v in payload.items()})
    if set(record) != {"literal", "value"}:
        raise ValueError("invalid program literal")
    kind, value = record["literal"], record["value"]
    if kind == "c-value-declaration":
        from .c_workflow import CWorkflow
        if type(value) is not list or len(value)!=2 or type(value[0]) is not str or type(value[1]) is not int:
            raise ValueError("invalid prospective C value declaration")
        return CWorkflow.from_bytes(base64.b64decode(value[0],validate=True)).value(value[1])
    if kind == "native-declaration":
        from .declaration_codec import declaration_types
        if (type(value) is not list or len(value) != 2 or type(value[0]) is not str
                or value[0] not in declaration_types() or type(value[1]) is not str):
            raise ValueError("unknown finite native declaration")
        return declaration_types()[value[0]](base64.b64decode(value[1], validate=True))
    if kind == "NoneType" and value is None:
        return None
    if kind == "bool" and type(value) is bool:
        return value
    if kind == "str" and type(value) is str:
        return value
    if kind == "int" and type(value) is str:
        return int(value, 16)
    if kind == "rational" and isinstance(value, list) and len(value) == 2:
        return Fraction(int(value[0], 16), int(value[1], 16))
    if kind == "float" and type(value) is str:
        result = float.fromhex(value)
        if math.isfinite(result):
            return result
    if kind == "bytes" and type(value) is str:
        return base64.b64decode(value, validate=True)
    if kind in {"tuple", "list"} and type(value) is list:
        result = [decode(v, depth+1) for v in value]
        return tuple(result) if kind == "tuple" else result
    if kind == "dict" and type(value) is list:
        out = {}
        for pair in value:
            if not isinstance(pair, list) or len(pair) != 2 or type(pair[0]) is not str or pair[0] in out:
                raise ValueError("invalid or repeated program record key")
            out[pair[0]] = decode(pair[1], depth+1)
        return out
    raise ValueError("unsupported program literal")


def dumps(value):
    return json.dumps(encode(value), sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode("utf8")


def loads(raw):
    if not isinstance(raw, (bytes, str)):
        raise TypeError("program data must be bytes or text")
    if len(raw) > 4*1024*1024:
        raise ValueError("program declaration exceeds the size limit")
    def pairs(items):
        result = {}
        for k, v in items:
            if k in result:
                raise ValueError("duplicate encoded field")
            result[k] = v
        return result
    try:
        result = decode(json.loads(raw, object_pairs_hook=pairs))
        if dumps(result) != (raw.encode("utf8") if isinstance(raw, str) else raw):
            raise ValueError("program declaration is not canonical")
        return result
    except (KeyError, TypeError, RecursionError, UnicodeError, json.JSONDecodeError, ZeroDivisionError) as exc:
        raise ValueError("invalid finite program data") from exc


def digest(value):
    return hashlib.sha256(dumps(value)).hexdigest()


def operators(query):
    found = set()
    def walk(value):
        if isinstance(value, ast.Call):
            found.add(value.name)
        if type(value).__name__ in _NODES and type(value) is _NODES[type(value).__name__]:
            for f in fields(value):
                walk(getattr(value, f.name))
        elif isinstance(value, (tuple, list)):
            for item in value:
                walk(item)
    for expression in (*query.bindings, *query.returns, *query.matches,
                       *((query.where,) if query.where is not None else ()), *query.order):
        walk(expression)
    return tuple(sorted(found))


def _implementation_inputs():
    import importlib.util
    from pathlib import Path
    files = []
    for package in ("rexgraph", "rcql", "rcdb", "rexgraph_temporal"):
        spec = importlib.util.find_spec(package)
        if spec is None and package == "rexgraph_temporal":
            continue
        if spec is None or spec.origin is None:
            raise RuntimeError("program implementation package is not installed")
        root = Path(spec.origin).parent
        for path in sorted(root.rglob("*")):
            if path.suffix not in {".py", ".so", ".pyd"} or "tests" in path.parts:
                continue
            name = (package+"/"+path.relative_to(root).as_posix()).encode()
            files.append((name, path))
    return tuple(files), _implementation_versions()


def _implementation_versions():
    import importlib.metadata
    versions = []
    for package in ("numpy", "scipy", "torch", "rdkit"):
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            version = "absent"
        versions.append((package+"="+version).encode())
    return tuple(versions)


def _implementation_digest(inputs):
    h = hashlib.sha256()
    files, versions = inputs
    for name, path in files:
        h.update(len(name).to_bytes(8, "big")); h.update(name)
        h.update(hashlib.sha256(path.read_bytes()).digest())
    for version in versions:
        h.update(version)
    return h.hexdigest()


def implementation_identity():
    """Identify installed Python definitions and compiled native modules."""
    return _implementation_digest(_implementation_inputs())


class ImplementationSnapshot:
    """A measured implementation, invalidated when its file inventory changes.

    The digest is the ordinary implementation identity. File seals avoid hashing
    unchanged shared libraries repeatedly; a changed seal triggers a full hash.
    This is a process-local optimization, never a replacement archive identity.
    """

    @staticmethod
    def _seal(inputs):
        files, versions = inputs
        rows = []
        for name, path in files:
            stat = path.stat()
            rows.append((name, str(path), stat.st_dev, stat.st_ino, stat.st_size,
                         stat.st_mtime_ns, stat.st_ctime_ns))
        return tuple(rows), versions

    def __init__(self):
        inputs = _implementation_inputs()
        before = self._seal(inputs)
        self.identity = _implementation_digest(inputs)
        if self._seal(_implementation_inputs()) != before:
            raise ValueError("implementation changed during preparation")
        self._stamp = before
        self._inputs = inputs
        self._directories = self._directory_inventory(inputs)
        self._fast_stamp = self._fast_seal()
        if self._seal(_implementation_inputs()) != before:
            raise ValueError("implementation changed during preparation")

    @staticmethod
    def _directory_inventory(inputs):
        from pathlib import Path
        roots = {path.parents[max(0, len(Path(name.decode()).parts) - 2)]
                 for name, path in inputs[0]}
        return tuple(sorted({root for root in roots}
            | {path for root in roots for path in root.rglob("*") if path.is_dir()}))

    def _fast_seal(self):
        inputs = self._inputs
        rows = self._seal((inputs[0], _implementation_versions()))
        directories = []
        for path in self._directories:
            stat = path.stat()
            directories.append((str(path), stat.st_dev, stat.st_ino, stat.st_mtime_ns, stat.st_ctime_ns))
        return rows, tuple(directories)

    def check(self):
        try:
            fast_stamp = self._fast_seal()
        except OSError as exc:
            raise ValueError("prepared implementation changed; create a new preparation") from exc
        if fast_stamp == self._fast_stamp:
            return self.identity
        inputs = _implementation_inputs()
        stamp = self._seal(inputs)
        if stamp != self._stamp:
            if (_implementation_digest(inputs) != self.identity
                    or self._seal(_implementation_inputs()) != stamp):
                raise ValueError("prepared implementation changed; create a new preparation")
            self._stamp = stamp
        self._inputs = inputs
        self._directories = self._directory_inventory(inputs)
        self._fast_stamp = self._fast_seal()
        if self._seal(_implementation_inputs()) != stamp:
            raise ValueError("prepared implementation changed; create a new preparation")
        return self.identity
