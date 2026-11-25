"""Importing this package registers every method.

One file per family.  A student adds a file (e.g. `components.py`), registers
its classes with `@register_detector` / `@register_steerer`, imports it here,
and lists the names in the config.
"""
from conflict_bench.methods import baselines, contrastive, representations  # noqa: F401
