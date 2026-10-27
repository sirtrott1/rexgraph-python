"""Closed data declarations; decoding never executes a native kernel."""
from functools import lru_cache


@lru_cache(maxsize=1)
def declaration_types():
    from .layer_product import LAYER_VALUES
    from .native_temporal import TemporalModel
    from .c_workflow import CWorkflow,CInterface
    return {**LAYER_VALUES,"TemporalModel":TemporalModel,"CWorkflow":CWorkflow,"CInterface":CInterface}
