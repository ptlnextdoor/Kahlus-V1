"""KTM A100 evidence intake auditor (read-only, SYNTHETIC lane).

Isolated consumer of a returned KTM A100 evidence folder/zip. Makes no scientific claim; it only
reports whether a bundle is complete, secret-safe, GPU-consistent, and claim-safe.
"""

from __future__ import annotations

from neurotwin.a100_audit.auditor import (
    AuditResult,
    Finding,
    audit_evidence,
    render_report_md,
)

__all__ = ["AuditResult", "Finding", "audit_evidence", "render_report_md"]
