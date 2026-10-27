"""Warehouse dialects use the same SQL catalog and native import operations."""
from .sql import SQLConnector


class WarehouseConnector(SQLConnector):
    pass
