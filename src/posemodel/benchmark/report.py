"""Self-contained benchmark report rendering."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


def _value(candidate: dict[str, Any], *path: str) -> str:
    value: Any = candidate
    for part in path:
        if not isinstance(value, dict) or part not in value:
            return "—"
        value = value[part]
    if isinstance(value, float):
        return f"{value:.4f}"
    return html.escape(str(value))


def write_report(report: dict[str, Any], directory: str | Path) -> tuple[Path, Path]:
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "benchmark.json"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    rows = []
    for name, candidate in report["candidates"].items():
        rows.append(
            "<tr>"
            f"<td>{html.escape(name)}</td>"
            f"<td>{_value(candidate, 'kind')}</td>"
            f"<td>{_value(candidate, 'evaluation', 'representation', 'dimensions')}</td>"
            f"<td>{_value(candidate, 'evaluation', 'supervised', 'macro_f1')}</td>"
            f"<td>{_value(candidate, 'evaluation', 'supervised', 'balanced_accuracy')}</td>"
            f"<td>{_value(candidate, 'evaluation', 'clustering', 'label_ami')}</td>"
            f"<td>{_value(candidate, 'evaluation', 'clustering', 'seed_stability_ami')}</td>"
            f"<td>{_value(candidate, 'fit_seconds')}</td>"
            f"<td>{_value(candidate, 'test_embedding_seconds')}</td>"
            "</tr>"
        )
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(report["name"])} — PoseModel benchmark</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 2rem auto; max-width: 1200px; color: #17202a; }}
h1, h2 {{ color: #102a43; }}
table {{ border-collapse: collapse; width: 100%; margin: 1rem 0 2rem; }}
th, td {{ border: 1px solid #bcccdc; padding: .55rem; text-align: right; }}
th:first-child, td:first-child {{ text-align: left; }}
th {{ background: #eaf2f8; }}
pre {{ background: #f3f6f8; padding: 1rem; overflow: auto; border-radius: .4rem; }}
.note {{ background: #fff8e1; padding: 1rem; border-left: 4px solid #ffb300; }}
</style>
</head>
<body>
<h1>{html.escape(report["name"])}</h1>
<p class="note">All normalization and representation fitting used the training split only.
Windows were constructed independently inside explicitly assigned recording groups.</p>
<h2>Dataset</h2>
<pre>{html.escape(json.dumps(report["dataset"], indent=2))}</pre>
<h2>Comparison</h2>
<table>
<thead><tr><th>Candidate</th><th>Kind</th><th>Dim</th><th>Macro F1</th>
<th>Balanced acc.</th><th>Label AMI</th><th>Seed stability</th><th>Fit s</th><th>Embed s</th></tr></thead>
<tbody>{"".join(rows)}</tbody>
</table>
<h2>Acceptance gates</h2>
<pre>{html.escape(json.dumps(report["acceptance_gates"], indent=2))}</pre>
<h2>Complete machine-readable results</h2>
<pre>{html.escape(json.dumps(report, indent=2))}</pre>
</body></html>"""
    html_path = output / "benchmark.html"
    html_path.write_text(document, encoding="utf-8")
    return json_path, html_path
