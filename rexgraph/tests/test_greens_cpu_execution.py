"""CSR execution retains channel trajectories and native optimizer continuation."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rexgraph.graph import RexGraph  # noqa: E402
from rexgraph.flow.cochain import CoParticipationCochain  # noqa: E402
from rexgraph.nn.optim import GreensCochain  # noqa: E402
from rexgraph.nn.lifecycle import capture_checkpoint, restore_checkpoint  # noqa: E402


class COOReference(GreensCochain):
    @staticmethod
    def _channel_op(adj, channel, group):
        power = GreensCochain._CH_POWER.get(channel, 1)
        cache = group.setdefault("_op_cache", {})
        if channel not in cache:
            op = adj
            for _ in range(power - 1):
                op = torch.sparse.mm(adj, op).coalesce()
            cache[channel] = op
        return cache[channel], channel != "high"

    @staticmethod
    def _execution_op(op, channel, group):
        return op

    @staticmethod
    def _greens(op, g, t, low, iters, power=1):
        return GreensCochain._greens(op, g, t, low, iters)


def adjacency(dtype):
    rows = torch.arange(12)
    indices = torch.stack((torch.cat((rows, rows, rows)),
                           torch.cat((rows, (rows + 1) % 12, (rows - 1) % 12))))
    return torch.sparse_coo_tensor(indices, torch.full((36,), 1.0 / 3, dtype=dtype),
                                   (12, 12), check_invariants=True).coalesce()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("channel", ["low", "high", "twohop", "threehop"])
def test_csr_and_coo_preserve_channel_updates(dtype, channel):
    generator = torch.Generator().manual_seed(37)
    initial = torch.randn(12, 3, generator=generator, dtype=dtype)
    target = torch.randn(12, 3, generator=generator, dtype=dtype)
    adj = adjacency(dtype)
    parameters = [torch.nn.Parameter(initial.clone()) for _ in range(2)]
    optimizers = [kind([{"params": [parameter], "green_adj": adj, "green_channel": channel,
                         "green_lam": 2.0, "green_iters": 12}], lr=0.05, weight_decay=0.01)
                  for kind, parameter in zip((GreensCochain, COOReference), parameters, strict=True)]
    for _ in range(20):
        for opt, parameter in zip(optimizers, parameters, strict=True):
            opt.zero_grad()
            (parameter - target).square().sum().backward()
            opt.step()
    torch.testing.assert_close(*parameters, rtol=1e-4 if dtype == torch.float32 else 1e-10,
                               atol=1e-5 if dtype == torch.float32 else 1e-11)
    group = optimizers[0].param_groups[0]
    assert group["green_adj"] is adj and adj.layout == torch.sparse_coo
    cached = group["_op_cache"][("cpu_csr", channel)][2]
    assert cached.layout == torch.sparse_csr
    assert cached._nnz() == adj._nnz()
    assert GreensCochain._execution_op(group["_op_cache"][channel], channel, group) is cached


def test_cpu_cache_tracks_mutated_values():
    adj = adjacency(torch.float64)
    group = {}
    before = GreensCochain._execution_op(adj, "low", group)
    adj.values().mul_(0.5)
    after = GreensCochain._execution_op(adj, "low", group)
    assert after is not before
    torch.testing.assert_close(after.to_dense(), adj.to_dense())


@pytest.mark.parametrize("channel", ["twohop", "threehop"])
def test_hop_channels_apply_only_sparse_dense_products(monkeypatch, channel):
    import rexgraph.nn.optim as module
    adj = adjacency(torch.float64)
    parameter = torch.nn.Parameter(torch.ones(12, 3, dtype=torch.float64))
    optimizer = GreensCochain([{"params": [parameter], "green_adj": adj,
                                "green_channel": channel}], green_iters=3)
    original = module._sparse_mm
    calls = []
    def checked(left, right):
        assert right.layout == torch.strided
        calls.append(left._nnz())
        return original(left, right)
    monkeypatch.setattr(module, "_sparse_mm", checked)
    parameter.square().sum().backward()
    optimizer.step()
    assert calls and set(calls) == {adj._nnz()}


def test_channel_cache_tracks_replaced_adjacency():
    group = {}
    original = adjacency(torch.float64)
    replacement = (original * 0.5).coalesce()
    assert GreensCochain._channel_op(original, "twohop", group)[0] is original
    assert GreensCochain._channel_op(replacement, "twohop", group)[0] is replacement


def test_native_checkpoint_discards_execution_cache_and_resumes():
    source = RexGraph(sources=np.arange(12, dtype=np.int32),
                      targets=(np.arange(12, dtype=np.int32) + 1) % 12)
    model = CoParticipationCochain(source, 3, dtype=torch.float64)
    optimizer = GreensCochain(model.greens_groups(), lr=0.1)
    target = torch.arange(36, dtype=torch.float64).reshape(12, 3) / 36
    def step(current, opt):
        opt.zero_grad()
        (current() - target).square().sum().backward()
        opt.step()
    step(model, optimizer)
    assert optimizer.param_groups[0]["_op_cache"][("cpu_csr", "low")][2].layout == torch.sparse_csr
    state = capture_checkpoint(model, source, configuration={"n_classes": 3, "dtype": "float64"},
                               optimizer=optimizer, optimizer_spec={"name": "auto", "lr": 0.1}, step=1)
    restored, resumed = restore_checkpoint(state, source, resume=True)
    step(model, optimizer)
    step(restored, resumed)
    torch.testing.assert_close(model.Z, restored.Z, rtol=0, atol=0)
    assert resumed.param_groups[0]["green_adj"].layout == torch.sparse_coo
