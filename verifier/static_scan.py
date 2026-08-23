from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
import re

from .schema import Finding

PROVEN = "PROVEN FAILURE"
HIGH = "HIGH-RISK / MANUAL REVIEW"
INFO = "INFO"


@dataclass(frozen=True)
class Risk:
    name: str
    line: int
    classification: str
    message: str


def _name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return ""


def _number(node: ast.AST) -> float | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value = _number(node.operand)
        return -value if value is not None else None
    return None


def _positive_offset(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.BinOp)
        and isinstance(node.op, ast.Add)
        and ((_number(node.left) or 0) > 0 or (_number(node.right) or 0) > 0)
    )


def _future_slice_upper(node: ast.AST) -> bool:
    """Python's ``: i + 1`` ends at the current row because stop is exclusive."""
    return (
        isinstance(node, ast.BinOp)
        and isinstance(node.op, ast.Add)
        and ((_number(node.left) or 0) > 1 or (_number(node.right) or 0) > 1)
    )


def _direct_future_slice_lower(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.BinOp)
        and isinstance(node.op, ast.Add)
        and (
            (isinstance(node.left, ast.Name) and (_number(node.right) or 0) > 0)
            or (isinstance(node.right, ast.Name) and (_number(node.left) or 0) > 0)
        )
    )


class LeakageVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.risks: list[Risk] = []

    def add(self, node: ast.AST, name: str, classification: str, message: str) -> None:
        self.risks.append(Risk(name, getattr(node, "lineno", 0), classification, message))

    def visit_Call(self, node: ast.Call) -> None:
        called = _name(node.func)
        attr = called.rsplit(".", 1)[-1]
        if attr == "shift" and node.args and (_number(node.args[0]) or 0) < 0:
            self.add(node, "negative_shift", HIGH, "negative shift exposes a future row; confirm it is label-only and never a causal feature")
        if called.endswith("np.roll") or called == "numpy.roll":
            shift = _number(node.args[1]) if len(node.args) > 1 else None
            shift = next((_number(k.value) for k in node.keywords if k.arg == "shift"), shift)
            if shift is not None and shift < 0:
                self.add(node, "negative_roll", HIGH, "negative np.roll can expose future observations")
        if attr == "rolling":
            center = next((k.value for k in node.keywords if k.arg == "center"), None)
            if isinstance(center, ast.Constant) and center.value is True:
                self.add(node, "centered_rolling", HIGH, "centered rolling window uses future observations")
        if attr in {"bfill", "backfill"}:
            self.add(node, "backfill", HIGH, "backfill can move future information backward")
        if attr == "fillna":
            rendered = ast.unparse(node) if hasattr(ast, "unparse") else ""
            if re.search(r"bfill|backfill", rendered, re.I):
                self.add(node, "fillna_bfill", HIGH, "fillna with backward fill can leak future values")
        if attr == "merge_asof":
            direction = next((k.value for k in node.keywords if k.arg == "direction"), None)
            if isinstance(direction, ast.Constant) and str(direction.value).lower() == "forward":
                self.add(node, "forward_asof", HIGH, "forward merge_asof selects future timestamps")
        if attr in {"sort_index", "sort_values"}:
            self.add(node, "silent_sort", HIGH, "sorting inside an engine may conceal corrupt input instead of rejecting it")
        if attr == "drop_duplicates":
            self.add(node, "drop_duplicates", HIGH, "deduplication inside an engine may conceal corrupt input")
        if attr == "interpolate":
            self.add(node, "interpolate", HIGH, "interpolation may synthesize absent market observations")
        if attr in {"fit", "fit_transform"}:
            self.add(node, "full_sample_fit", HIGH, "model/preprocessor fitting requires manual confirmation that fitting occurs only on the legal training sample")
        if attr in {"mean", "std", "quantile", "median"}:
            owner = called.rsplit(".", 1)[0] if "." in called else ""
            if not any(token in owner.lower() for token in ("rolling", "expanding", "groupby", "window")):
                self.add(node, "full_sample_statistic", INFO, "global statistic may be safe reporting or may leak if reused as a causal feature")
        if called == "reversed" or attr == "reverse":
            self.add(node, "reverse_iteration", INFO, "reverse iteration can implement future accumulation; inspect information flow")
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        target = _name(node.value)
        sub = node.slice
        if target.endswith("iloc") and _positive_offset(sub):
            self.add(node, "future_iloc", HIGH, "iloc indexed with a positive horizon (for example i+1)")
        if isinstance(sub, ast.Slice):
            if _direct_future_slice_lower(sub.lower) or _future_slice_upper(sub.upper):
                self.add(node, "future_slice", HIGH, "slice boundary uses a positive horizon")
            if (_number(sub.step) or 0) < 0:
                self.add(node, "reverse_slice", INFO, "reverse slice may support future accumulation")
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        names = " ".join(_name(t) for t in node.targets).lower()
        if any(token in names for token in ("future_return", "forward_return", "future_label", "target_return")):
            self.add(node, "future_label", INFO, "future-return label detected; prove it is isolated from causal features and execution")
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:
        rendered = ast.unparse(node) if hasattr(ast, "unparse") else ""
        if re.search(r"timestamp.*>|>.*timestamp", rendered, re.I):
            self.add(node, "future_timestamp_join", INFO, "timestamp-greater-than comparison may support a future join; inspect manually")
        self.generic_visit(node)


FALLBACK_PATTERNS = [
    ("negative_shift", r"\.shift\s*\(\s*-\d+", HIGH, "negative shift can import future values"),
    ("backfill", r"\.(?:bfill|backfill)\s*\(", HIGH, "backfill can move future information backward"),
    ("interpolate", r"\.interpolate\s*\(", HIGH, "interpolation may synthesize missing observations"),
]


def _scan_file(path: Path) -> list[Finding]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return [Finding("static_source", "WARN", f"could not read {path}: {exc}", family="static_scan", classification=HIGH)]
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError as exc:
        return [Finding("static_syntax", "FAIL", f"{path}:{exc.lineno}: {exc.msg}", family="static_scan", classification=PROVEN)]
    visitor = LeakageVisitor()
    visitor.visit(tree)
    risks = list(visitor.risks)
    # Regex fallback preserves detection in syntactically odd/generated source.
    seen = {(r.name, r.line) for r in risks}
    for name, pattern, classification, message in FALLBACK_PATTERNS:
        for match in re.finditer(pattern, text, re.I | re.S):
            line = text.count("\n", 0, match.start()) + 1
            if (name, line) not in seen:
                risks.append(Risk(name, line, classification, message))
    return [
        Finding(
            f"static_{risk.name}", "FAIL" if risk.classification == PROVEN else ("WARN" if risk.classification == HIGH else "INFO"),
            f"{path}: line {risk.line}: {risk.message}", family="static_scan", classification=risk.classification,
            evidence={"path": str(path), "line": risk.line},
        )
        for risk in sorted(risks, key=lambda r: (r.line, r.name))
    ]


def scan_source(path: str | Path) -> list[Finding]:
    p = Path(path)
    if not p.exists():
        return [Finding("static_source", "FAIL", f"source path not found: {p}", family="static_scan", classification=PROVEN)]
    paths = [p] if p.is_file() else sorted(p.rglob("*.py"))
    findings = [finding for source in paths for finding in _scan_file(source)]
    if not findings:
        findings.append(Finding(
            "static_scan", "PASS",
            "no configured AST/regex risk patterns found; a passing static scan does not prove absence of leakage",
            family="static_scan", classification=INFO,
        ))
    return findings
