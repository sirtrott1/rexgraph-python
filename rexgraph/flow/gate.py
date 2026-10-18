"""Detect changes in a stream of Malaugh topology entropy readings.

The gate compares the magnitude of each nonzero entropy change with the
median and median absolute deviation of earlier changes. This lets a small
change be anomalous after steady growth, just as a large change can be. A
cycle closure after leaf growth is one tested example, not a classification
rule for arbitrary source edits.

State consists of the previous entropy and all past changes exceeding eps.
The history grows with the stream, and recomputing its median and deviation
costs O(len(history)) per accepted change. No eigendecomposition or dense
operator is needed for the gate's scalar bookkeeping."""
from __future__ import annotations

import numpy as np

from rexgraph.scale_propagator import malaugh_quantities

__all__ = ["malaugh_entropy", "MalaughGate"]


def malaugh_entropy(rex) -> float:
    """Return topology harmonic log entropy H_T for one snapshot.

    The reading depends on T = B1.T B1 and is independent of faces. A zero trace
    makes the entropy undefined; a NaN result raises ValueError. The gate checks
    that a supplied entropy is finite before accepting it."""
    h_t = malaugh_quantities(rex)["H_T"]
    if h_t != h_t:  # nan check without importing math.isnan
        raise ValueError("malaugh_entropy: H_T is nan for this rex (empty or degenerate complex)")
    return float(h_t)


class MalaughGate:
    """Compare each nonzero entropy change with the stream's past changes.

    ``observe`` computes H_T and passes it to ``observe_entropy``. A change with
    magnitude at most ``eps`` updates the previous value but does not enter the
    baseline. After ``warmup`` accepted changes, an event occurs when the new
    magnitude differs from the prior median by more than ``fence_k`` times the
    prior median absolute deviation, floored at ``eps``.

    Both unusually small and unusually large changes can produce events. An
    event identifies a change in this scalar reading; interpreting its cause
    requires the source history. Use a separate instance for each stream so its
    baseline is not mixed with another stream's observations."""

    def __init__(self, fence_k: float = 3.0, warmup: int = 3, eps: float = 1e-9):
        self.fence_k = float(fence_k)
        self.warmup = int(warmup)
        self.eps = float(eps)
        self._prev: float | None = None
        self._hist: list[float] = []

    def observe(self, rex) -> dict[str, object]:
        h_t = malaugh_entropy(rex)
        return self.observe_entropy(h_t)

    def observe_entropy(self, h_t) -> dict[str, object]:
        """Apply the same gate to an already computed finite entropy value."""
        h_t = float(h_t)
        if not np.isfinite(h_t):
            raise ValueError("gate entropy must be finite")

        if self._prev is None:
            self._prev = h_t
            return {"H_T": h_t, "delta": 0.0, "event": False}

        delta = h_t - self._prev
        self._prev = h_t
        mag = abs(delta)

        event = False
        if mag > self.eps:
            if len(self._hist) >= self.warmup:
                median = float(np.median(self._hist))
                mad = float(np.median(np.abs(np.asarray(self._hist) - median)))
                mad = max(mad, self.eps)
                event = abs(mag - median) > self.fence_k * mad
            self._hist.append(mag)

        return {"H_T": h_t, "delta": float(delta), "event": bool(event)}
