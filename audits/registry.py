# audits/registry.py
from typing import Callable, Dict, Any

class Registry:
    def __init__(self):
        self.metric_fns: Dict[str, Callable[[dict], dict]] = {}
        self.validators: Dict[str, Callable[[dict], dict]] = {}

    def metric(self, name: str):
        def deco(fn: Callable[[dict], dict]):
            self.metric_fns[name] = fn
            return fn
        return deco

    def validator(self, name: str):
        def deco(fn: Callable[[dict], dict]):
            self.validators[name] = fn
            return fn
        return deco

plugin_registry = Registry()
