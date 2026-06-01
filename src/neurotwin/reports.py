"""Compatibility shim for report helpers now owned by neurotwin.benchmarks.

TODO: remove this module after external callers migrate to
neurotwin.benchmarks.reports.
"""

from neurotwin.benchmarks.reports import generate_compare_report, generate_run_report, generate_suite_report

__all__ = ["generate_compare_report", "generate_run_report", "generate_suite_report"]
