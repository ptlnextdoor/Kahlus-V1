from __future__ import annotations

from typing import Any

from neurotwin.reports.model_card import generate_model_card_report

__all__ = [
    "generate_compare_report",
    "generate_model_card_report",
    "generate_run_report",
    "generate_suite_report",
]


def generate_compare_report(*args: Any, **kwargs: Any) -> str:
    from neurotwin.benchmarks.reports import generate_compare_report as _generate_compare_report

    return _generate_compare_report(*args, **kwargs)


def generate_run_report(*args: Any, **kwargs: Any) -> str:
    from neurotwin.benchmarks.reports import generate_run_report as _generate_run_report

    return _generate_run_report(*args, **kwargs)


def generate_suite_report(*args: Any, **kwargs: Any) -> str:
    from neurotwin.benchmarks.reports import generate_suite_report as _generate_suite_report

    return _generate_suite_report(*args, **kwargs)
