"""A starting point for a structural connector to an external source.

Copy this file, replace the source reading and edge construction steps in
MyConnector, and declare the capabilities it actually emits. This template
uses pairwise links; richer relation declarations need a construction that
preserves their full participant sets.

From the repository root:

    cp agent/agent/connectors/template.py agent/agent/connectors/my_system.py

Validate the connector against a representative input:

    from agent.connectors.validate import validate_connector
    from agent.connectors.my_system import MyConnector
    print(validate_connector(MyConnector(), my_source))

The local example needs no external service:

    python -m agent.connectors.template"""

from __future__ import annotations

from typing import Any

from . import BaseConnector, Capabilities

# the skeleton to copy

class MyConnector(BaseConnector):
    """Rename me. Read a source read only and return ``(rex, meta)``."""

    # Advertise what you can supply. topology is always True; flip the others
    # on only if you actually emit them below. ``schemes`` are the URI schemes
    # the registry should route to you (e.g. ("mysystem",)).
    CAPABILITIES = Capabilities(
        weights=False,      # set True if you emit meta["weights"]
        modality=False,     # set True if you emit meta["modality"]
        faces=False,        # set True if you emit meta["faces"]
        schemes=(),         # e.g. ("mysystem",)
    )

    def read(self, source: Any) -> tuple[Any, dict[str, Any]]:
        # Replace with a read only source adapter. This structural connector
        # selects entities and links; record payload ingestion is a separate step.
        # Produce, from your source:
        #   labels : list[str]                 one per entity (vertex)
        #   links  : list[(src_label, dst_label)]   one per relationship (edge)
        labels: list[str] = []          # e.g. table / node / class names
        links: list[tuple[str, str]] = []   # e.g. child->parent FK pairs

        # Map the selected pairwise links to the vertex indices above.
        # Map each relationship to an edge between two vertices. B₁ is built for
        # you from these index pairs (source = -1, target = +1). Optionally:
        #   weights  : list[float]   per edge, with an application defined meaning
        #   modality : list[dict]    per edge: {"nullable":..,"identifying":..}
        #   faces    : np.ndarray    dense B₂ (nE×nF) if some edges co close
        idx = {name: i for i, name in enumerate(labels)}
        sources = [idx[a] for a, _ in links]
        targets = [idx[b] for _, b in links]

        return self.result(
            (sources, targets),
            vertex_labels=labels,
            edges=links,
            source="mysystem://<identifier>",
            # weights=[...],           # if CAPABILITIES.weights
            # modality=[...],          # if CAPABILITIES.modality
            # faces=b2_dense,          # if CAPABILITIES.faces
        )


# a worked example that runs immediately

class ExampleEdgesConnector(BaseConnector):
    """Read weighted in memory links as a small structural source.

    Weights are the supplied cardinalities. Their analysis follows that chosen
    interpretation; the connector does not infer a physical metric from them."""

    CAPABILITIES = Capabilities(weights=True, schemes=("example",))

    def read(self, source: Any) -> tuple[Any, dict[str, Any]]:
        # ``source`` here is a list of (parent, child, cardinality) triples -
        # e.g. a hand written schema. A real connector reads this from the
        # system instead of taking it as an argument.
        rows = source or [
            ("orders", "customers", 1000),
            ("order_items", "orders", 5000),
            ("order_items", "products", 5000),
            ("payments", "orders", 1200),
        ]
        labels: list[str] = []
        for parent, child, _ in rows:
            for name in (parent, child):
                if name not in labels:
                    labels.append(name)
        idx = {n: i for i, n in enumerate(labels)}
        links = [(a, b) for a, b, _ in rows]
        sources = [idx[a] for a, b, _ in rows]
        targets = [idx[b] for a, b, _ in rows]
        weights = [float(w) for _, _, w in rows]
        return self.result(
            (sources, targets),
            vertex_labels=labels,
            edges=links,
            weights=weights,
            source="example://mini-schema",
        )


def _demo() -> None:
    rex, meta = ExampleEdgesConnector().read(None)
    print("capabilities:", ExampleEdgesConnector().capabilities().summary())
    print("nV=%d nE=%d source=%s" % (meta["nV"], meta["nE"], meta["source"]))
    print("labels:", meta["vertex_labels"])
    print("edges :", meta["edges"])
    print("weights:", meta["weights"])
    from . import to_rexgraph
    g = to_rexgraph(rex, meta)
    print("built complex: nV=%d nE=%d betti=%s chain_valid=%s"
          % (g.nV, g.nE, tuple(g.betti), g.chain_valid))


if __name__ == "__main__":
    _demo()
