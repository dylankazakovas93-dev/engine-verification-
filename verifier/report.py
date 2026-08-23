from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from .schema import Finding

VERIFIED = "VERIFIED"
FAILED = "FAILED"
INCOMPLETE = "INCOMPLETE / UNVERIFIED"
EXIT_CODES = {VERIFIED: 0, FAILED: 1, INCOMPLETE: 2}


def _family(finding: Finding) -> str:
    return finding.family or finding.check


def evaluate_verdict(findings: list[Finding], mandatory_checks: set[str] | None = None) -> str:
    """Return a machine gate; missing mandatory evidence can never become VERIFIED."""
    mandatory = set(mandatory_checks or ())
    if not mandatory:
        if any(f.status == "FAIL" for f in findings):
            return FAILED
        return INCOMPLETE if any(f.status in {"UNVERIFIED", "WARN"} for f in findings) else VERIFIED
    related_by_check = {
        name: [f for f in findings if _family(f) == name or f.check == name]
        for name in mandatory
    }
    if any(
        f.status == "FAIL"
        for related in related_by_check.values()
        for f in related
    ):
        return FAILED
    for name in mandatory:
        related = related_by_check[name]
        if not related:
            return INCOMPLETE
        if any(f.status in {"UNVERIFIED", "WARN"} for f in related):
            return INCOMPLETE
        if not any(f.status in {"PASS", "INFO"} for f in related):
            return INCOMPLETE
    return VERIFIED


@dataclass
class VerificationReport:
    strategy: str
    candidate: str
    findings: list[Finding] = field(default_factory=list)
    mandatory_checks: set[str] = field(default_factory=set)
    artifacts: dict[str, Any] = field(default_factory=dict)
    coverage: dict[str, Any] = field(default_factory=dict)
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def verdict(self) -> str:
        return evaluate_verdict(self.findings, self.mandatory_checks)

    @property
    def exit_code(self) -> int:
        return EXIT_CODES[self.verdict]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "generated_at": self.generated_at,
            "strategy": self.strategy,
            "candidate": self.candidate,
            "verdict": self.verdict,
            "exit_code": self.exit_code,
            "mandatory_checks": sorted(self.mandatory_checks),
            "coverage": self.coverage,
            "artifacts": self.artifacts,
            "findings": [f.to_dict() for f in self.findings],
        }

    def write_json(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2, default=str) + "\n", encoding="utf-8")
        return target

    def write_markdown(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            f"# Verification Report — {self.strategy}",
            "",
            f"- Verdict: **{self.verdict}**",
            f"- Candidate: `{self.candidate}`",
            f"- Generated: `{self.generated_at}`",
            f"- Mandatory checks: {', '.join(sorted(self.mandatory_checks)) or 'none configured'}",
            "",
            "## Findings",
            "",
            "| Status | Family | Check | Classification | Result |",
            "|---|---|---|---|---|",
        ]
        for f in self.findings:
            msg = f.message.replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {f.status} | {_family(f)} | {f.check} | {f.classification or ''} | {msg} |")
        if self.coverage:
            lines += ["", "## Coverage", "", "```json", json.dumps(self.coverage, indent=2, default=str), "```"]
        if self.artifacts:
            lines += ["", "## Artifacts", "", "```json", json.dumps(self.artifacts, indent=2, default=str), "```"]
        target.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return target


def print_report(findings: list[Finding], mandatory_checks: set[str] | None = None) -> int:
    counts = Counter(x.status for x in findings)
    verdict = evaluate_verdict(findings, mandatory_checks)
    print("\n=== BACKTEST VERIFICATION REPORT ===")
    for x in findings:
        classification = f" [{x.classification}]" if x.classification else ""
        print(f"[{x.status:10}] {x.check}{classification}: {x.message}")
        if x.rows:
            print(f"             rows: {x.rows}")
    print("\nSummary:", " ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print("VERDICT:", verdict)
    return EXIT_CODES[verdict]
