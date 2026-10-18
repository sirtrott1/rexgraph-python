"""Owning algebraic records retain exact coordinates through execution and storage."""
import gc
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from rexgraph_cpp import NativeAlgebraicArchive, NativeAlgebraicValue

from rexgraph.io.algebraic_record import algebraic_record, read_algebraic_record

_ARCHIVE_PROFILES = ("rational", "real_cubic", "complex_quadratic", "complex_quartic", "complex_cyclotomic")
_complete_cases = pytest.mark.parametrize("archive_case", _ARCHIVE_PROFILES, indirect=True)

@pytest.fixture(scope="session")
def algebraic_fixtures(tmp_path_factory):
    folder = tmp_path_factory.mktemp("algebraic-fixtures")
    executable = os.environ.get("REXGRAPH_CPP_ALGEBRAIC_FIXTURE")
    if executable is None:
        build = folder / "build"
        subprocess.run(["cmake", "-S", str(Path(__file__).resolve().parents[1]), "-B", str(build),
                        "-DREXGRAPH_BINDING_TEST_FIXTURE=ON", f"-DPython3_EXECUTABLE={sys.executable}"],
                       check=True, capture_output=True, text=True)
        subprocess.run(["cmake", "--build", str(build), "--target", "algebraic_fixture", "-j1"],
                       check=True, capture_output=True, text=True)
        executable = build / "algebraic_fixture"
    result = subprocess.run([str(executable), str(folder)], check=True, capture_output=True, text=True)
    return folder, {row["profile"]: row for row in map(json.loads, result.stdout.splitlines())}


@pytest.fixture(params=_ARCHIVE_PROFILES[:4])
def archive_case(request, algebraic_fixtures):
    folder, oracle = algebraic_fixtures
    raw = (folder / (request.param + ".rfcxa")).read_bytes()
    return NativeAlgebraicArchive(raw), oracle[request.param]


def exact_rows(values):
    return tuple(tuple(Fraction(v) for v in row) for row in values)


@_complete_cases
def test_archive_manifest_retains_declared_field_and_source(archive_case):
    archive, oracle = archive_case
    manifest = archive.manifest
    assert manifest["codec"] == "rfcxa-v1"
    assert manifest["field_degree"] == {"rational": 1, "real_cubic": 3,
                                         "complex_quadratic": 2, "complex_quartic": 4,
                                         "complex_cyclotomic": 4}[oracle["profile"]]
    assert manifest["source_sizes"] == (3, 3, 2, 0)
    assert manifest["flat"] and manifest["chain_policy"] == "require_flat"
    assert manifest["metric_domain"] and manifest["reconstructions"] == (100,)
    assert dict(manifest["records"])[80] == "form"
    assert dict(manifest["records"])[50] == "metric"
    assert all(isinstance(q, Fraction) for q in manifest["minimal_polynomial"])
    assert archive.record(90).shape == (3, 2) and archive.record(91).shape == (2, 3)
    assert archive.record(10).preview(0)["source_carrier"]["rank"] == 2
    certificates = {"rational": {"kind":"irreducible_reduction", "prime":3},
                    "real_cubic": {"kind":"eisenstein_shift", "prime":2, "shift":0},
                    "complex_quadratic": {"kind":"irreducible_reduction", "prime":3},
                    "complex_quartic": {"kind":"eisenstein_shift", "prime":2, "shift":1},
                    "complex_cyclotomic": {"kind":"cyclotomic", "order":12}}
    assert manifest["certificate"] == certificates[oracle["profile"]]
    assert NativeAlgebraicArchive(archive.archive).manifest == manifest


@_complete_cases
def test_exact_operations_match_native_reference(archive_case):
    archive, oracle = archive_case
    x, metric = archive.record(10), archive.record(50)
    results = {"map": archive.record(30).apply(x), "solve": metric.solve(x),
               "quadrance": metric.quadrance(x), "covector": metric.as_covector(x),
               "evaluate": archive.record(20).evaluate(x), "accession": archive.accession(100, x)}
    results["reconstruct"] = archive.reconstruct(100, results["accession"])
    for name, value in results.items():
        assert value.coefficients == exact_rows(oracle[name])
        assert value.archive_owner is archive
    assert metric.apply(results["solve"]).coefficients == x.coefficients
    assert results["covector"].evaluate(x).coefficients == results["quadrance"].coefficients
    assert results["quadrance"].kind == "scalar" and results["quadrance"].shape == ()


def test_rectangular_reconstruction_keeps_only_its_declared_left_inverse(archive_case):
    archive, _ = archive_case
    x = archive.record(10)
    image = archive.accession(100, x)
    projected = archive.project(100, image)
    assert projected.shape == (3, 1)
    assert projected.coefficients == image.coefficients
    assert archive.project(100, projected).coefficients == projected.coefficients
    assert archive.reconstruct(100, image).coefficients == x.coefficients
    with pytest.raises(ValueError, match="owner"):
        archive.project(100, x)


def test_equal_archive_bytes_do_not_identify_live_carriers(archive_case):
    archive, _ = archive_case
    other = NativeAlgebraicArchive(archive.archive)
    assert other.manifest == archive.manifest and other is not archive
    for key in (10, 11):
        x = other.record(key)
        for action in (archive.record(30).apply, archive.record(50).solve,
                       archive.record(20).evaluate, lambda value: archive.accession(100, value)):
            with pytest.raises(ValueError, match="owner"):
                action(x)


def test_supported_zero_and_record_kinds_are_preserved(archive_case):
    archive, _ = archive_case
    zero = archive.record(11)
    assert len(zero.coefficients) == 2
    assert all(q == 0 for row in zero.coefficients for q in row)
    assert archive.record(50).solve(zero).coefficients == zero.coefficients
    for action in (archive.record(80).solve, archive.record(30).quadrance, zero.apply):
        with pytest.raises(ValueError, match="not defined"):
            action(zero)
    with pytest.raises(ValueError, match="module vector"):
        archive.record(30).apply(archive.record(20))
    with pytest.raises(TypeError):
        archive.record(30).apply(zero.coefficients)


@_complete_cases
def test_owners_and_exact_values_survive_local_reference_release(archive_case):
    archive, _ = archive_case
    raw = archive.archive
    result = archive.record(50).solve(archive.record(10))
    del archive
    gc.collect()
    owner = result.archive_owner
    assert owner.archive == raw
    assert owner.record(50).apply(result).coefficients == owner.record(10).coefficients


def test_parallel_operations_retain_the_same_exact_result(archive_case):
    archive, _ = archive_case
    metric, value = archive.record(50), archive.record(10)
    expected = metric.solve(value).coefficients
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = tuple(pool.map(lambda _: metric.solve(value).coefficients, range(24)))
    assert all(v == expected for v in results)


def test_named_reconstruction_preparation_retains_owners_and_parallel_exactness(archive_case):
    archive, _ = archive_case
    value = archive.record(10)
    expected = archive.accession(100, value).coefficients
    foreign = NativeAlgebraicArchive(archive.archive).record(10)
    with pytest.raises(ValueError):
        archive.accession(100, foreign)
    with pytest.raises((ValueError, IndexError)):
        archive.accession(999, value)
    def reconstruct(_):
        physical = archive.accession(100, value)
        assert physical.coefficients == expected
        return archive.reconstruct(100, physical).coefficients
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = tuple(pool.map(reconstruct, range(24)))
    assert all(result == value.coefficients for result in results)


@pytest.mark.parametrize("limit,value", [("max_records", 1), ("max_rational_coordinates", 0),
    ("max_integer_bytes", 1), ("field.max_degree", 0), ("field.max_scalar_steps", 1),
    ("modules.max_cells", 1), ("modules.max_rank", 1), ("fibers.max_occurrences", 1),
    ("metric.max_factor_steps", 1)])
def test_each_native_admission_limit_is_enforced(archive_case, limit, value):
    archive, _ = archive_case
    with pytest.raises((ValueError, OverflowError, RuntimeError)):
        NativeAlgebraicArchive(archive.archive, limits={limit: value})


def test_archive_byte_admission_is_exact(archive_case):
    archive, _ = archive_case
    size = len(archive.archive)
    assert NativeAlgebraicArchive(archive.archive, limits={"max_file_bytes": size}).archive == archive.archive
    with pytest.raises(ValueError):
        NativeAlgebraicArchive(archive.archive, limits={"max_file_bytes": size - 1})


def test_field_degree_admission_accepts_an_explicit_larger_bound(archive_case):
    archive, _ = archive_case
    limits = {"field.max_degree": 64}
    loaded = NativeAlgebraicArchive(archive.archive, limits=limits)
    restored = read_algebraic_record(algebraic_record(archive), limits=limits)
    assert loaded.limits["field.max_degree"] == 64
    assert loaded.archive == restored.archive == archive.archive
    assert loaded.manifest["field_degree"] == archive.manifest["field_degree"]


@pytest.mark.parametrize("value", [True, -1, 1.5, float("inf"), 1 << 100])
def test_admission_fields_do_not_coerce_invalid_values(algebraic_fixtures, value):
    raw = (algebraic_fixtures[0] / "rational.rfcxa").read_bytes()
    with pytest.raises((TypeError, ValueError, OverflowError)):
        NativeAlgebraicArchive(raw, limits={"max_file_bytes": value})


@pytest.mark.parametrize("key", [True, -1, 1.5, 1 << 100, 999])
def test_record_names_require_declared_integer_keys(algebraic_fixtures, key):
    archive = NativeAlgebraicArchive((algebraic_fixtures[0] / "rational.rfcxa").read_bytes())
    with pytest.raises((TypeError, ValueError, OverflowError, IndexError)):
        archive.record(key)


@_complete_cases
def test_archive_and_manifests_are_immutable(archive_case):
    archive, _ = archive_case
    expected = archive.manifest
    changed = archive.manifest
    changed["certificate"]["prime"] = 0
    changed["embedding"]["kind"] = "other"
    assert archive.manifest == expected
    limits = archive.limits
    limits["max_file_bytes"] = 0
    assert archive.limits["max_file_bytes"] > 0
    with pytest.raises(RuntimeError, match="immutable"):
        archive.__init__(archive.archive)


@pytest.mark.parametrize("cls", [NativeAlgebraicArchive, NativeAlgebraicValue])
def test_uninitialized_native_handles_reject_access(cls):
    value = cls.__new__(cls)
    with pytest.raises(RuntimeError, match="not initialized"):
        value.preview()


@pytest.mark.parametrize("change", ["archive", "source_digest", "archive_digest", "shape", "dtype", "version", "extra"])
@_complete_cases
def test_attachments_reject_current_corruption(archive_case, change):
    archive, _ = archive_case
    record = algebraic_record(archive)
    value = record.get_metadata(1, 0, "algebraic_archive")
    if change == "archive":
        value["archive"][-1] ^= 1
    elif change in {"source_digest", "archive_digest"}:
        value[change] = "f" * 64
    elif change == "shape":
        value["archive"] = value["archive"].reshape(1, -1)
    elif change == "dtype":
        value["archive"] = value["archive"].astype(np.int64)
    elif change == "version":
        value["version"] = True
    else:
        value["extra"] = True
    with pytest.raises(ValueError):
        read_algebraic_record(record)


def test_record_restore_keeps_bytes_but_creates_fresh_owners(archive_case):
    from rexgraph.state import from_state, to_state
    archive, _ = archive_case
    record = from_state(to_state(algebraic_record(archive)))
    restored = read_algebraic_record(record)
    assert restored.archive == archive.archive and restored.manifest == archive.manifest
    assert restored.record(10).coefficients == archive.record(10).coefficients
    with pytest.raises(ValueError, match="owner"):
        restored.record(50).solve(archive.record(10))
    with pytest.raises(ValueError):
        read_algebraic_record(record, limits={"max_file_bytes": 0})


@pytest.mark.parametrize("suffix", [".rcbd", ".safetensors", ".h5", ".zarr", ".json"])
@_complete_cases
def test_core_containers_keep_complete_archive_bytes(archive_case, tmp_path, suffix):
    from rexgraph.io import load, save
    dependency = {".safetensors": "safetensors", ".h5": "h5py", ".zarr": "zarr"}.get(suffix)
    if dependency is not None:
        pytest.importorskip(dependency)
    archive, _ = archive_case
    path = tmp_path / ("archive" + suffix)
    save(path, algebraic_record(archive))
    restored = read_algebraic_record(load(path))
    assert restored.archive == archive.archive
    assert restored.record(50).solve(restored.record(10)).coefficients == archive.record(50).solve(archive.record(10)).coefficients


@_complete_cases
def test_agent_and_system_retain_exact_owning_values(archive_case):
    from agent.hive import Hive
    from system.serialize import json_value
    archive, _ = archive_case
    hive = Hive("algebraic-fixture")
    hive.add_worker("exact-map", archive.record(30).apply, capability="predict")
    x = archive.record(10)
    result = hive.invoke("exact-map", x, record=False)
    assert result.coefficients == archive.record(30).apply(x).coefficients
    view = json_value(result, max_values=1)
    assert view["kind"] == "NativeAlgebraicValue" and view["value_kind"] == "vector"
    assert view["truncated"] and len(view["coefficients"]) == 1
    assert json_value(archive, max_values=0)["records"] == []
    assert json_value(result, max_values=0)["coefficients"] == []
    assert len(json.dumps(view, allow_nan=False)) > 0


@pytest.mark.parametrize("backend", ["local", "file", "rex", "sqlite"])
@_complete_cases
def test_rcdb_cold_restore_retains_archive_and_exact_execution(archive_case, tmp_path, backend):
    from rcdb import open_store
    archive, _ = archive_case
    location = tmp_path / "store"
    uri = f"sqlite:///{location}.sqlite" if backend == "sqlite" else f"{backend}://{location}"
    store = open_store(uri, **({"read_only": False} if backend in {"file", "rex"} else {}))
    store.configure_security(require_commits=True)
    try:
        store.commit_mutation("algebraic", algebraic_record(archive), expected_version=0)
        assert store.verify_commits("algebraic")
    finally:
        store.close()
    code = """
import json,sys
import rexgraph,rexgraph_cpp,rexgraph.core._laplacians as compiled
from rcdb import open_store
from rexgraph.io.algebraic_record import read_algebraic_record
store=open_store(sys.argv[1]).configure_security(require_commits=True)
try:
    assert store.verify_commits('algebraic')
    archive=read_algebraic_record(store.read_record('algebraic',version=1).value)
    x=archive.record(10);metric=archive.record(50)
    assert metric.apply(metric.solve(x)).coefficients==x.coefficients
    print(json.dumps({'digest':archive.manifest['archive_digest'],
                      'solved':[[str(q) for q in row] for row in metric.solve(x).coefficients],
                      'core_source':rexgraph.__file__,'binding_source':rexgraph_cpp.__file__,
                      'compiled_core':compiled.__file__}))
finally:store.close()
"""
    result = subprocess.run([sys.executable, "-I", "-c", code, uri], cwd=tmp_path,
                            check=True, capture_output=True, text=True)
    returned = json.loads(result.stdout)
    assert returned["digest"] == archive.manifest["archive_digest"]
    assert exact_rows(returned["solved"]) == archive.record(50).solve(archive.record(10)).coefficients
    import rexgraph
    assert Path(returned["core_source"]).resolve() == Path(rexgraph.__file__).resolve()
    assert str(Path(sys.prefix) / "lib") in returned["binding_source"]


@pytest.mark.parametrize("index", [True, -1, 1.5])
def test_record_selection_requires_nonnegative_integer_cells(algebraic_fixtures, index):
    archive = NativeAlgebraicArchive((algebraic_fixtures[0] / "rational.rfcxa").read_bytes())
    with pytest.raises(ValueError):
        read_algebraic_record(algebraic_record(archive), index)


def test_unknown_resource_names_and_archive_corruption_reject(algebraic_fixtures):
    raw = (algebraic_fixtures[0] / "rational.rfcxa").read_bytes()
    with pytest.raises(ValueError, match="admission"):
        NativeAlgebraicArchive(raw, limits={"other": 100})
    for bytes_value in (raw[:16], raw[:-1], raw[:-1] + bytes([raw[-1] ^ 1])):
        with pytest.raises((ValueError, RuntimeError)):
            NativeAlgebraicArchive(bytes_value)


def test_missing_structures_are_not_inferred(algebraic_fixtures):
    archive = NativeAlgebraicArchive((algebraic_fixtures[0] / "plain.rfcxa").read_bytes())
    manifest = archive.manifest
    assert manifest["involution"] is None and manifest["embedding"] is None
    assert not manifest["metric_domain"]
    assert manifest["chain_policy"] is None and manifest["flat"] is None
    assert manifest["records"] == ((10, "vector"),)


def test_declared_deformation_retains_its_nonflat_policy(algebraic_fixtures):
    archive = NativeAlgebraicArchive((algebraic_fixtures[0] / "deformed.rfcxa").read_bytes())
    assert archive.manifest["chain_policy"] == "declared_deformation"
    assert archive.manifest["flat"] is False
    restored = read_algebraic_record(algebraic_record(archive))
    assert restored.archive == archive.archive and restored.manifest == archive.manifest
