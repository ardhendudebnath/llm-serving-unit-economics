"""Measurement and cost tooling for one served open-weight model.

The core -- config, metrics, workloads, cost -- is stdlib-only. Only the load
generator and the chart renderer take dependencies, and both are optional
extras. See `pyproject.toml` for why.
"""
