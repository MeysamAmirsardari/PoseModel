# Contributing

Use Python 3.10 or newer and install the development extras:

```bash
python -m pip install -e '.[dev]'
ruff check .
ruff format --check .
pytest
```

New model claims must include a leakage-safe comparison against the baselines documented in
`docs/architecture.md`. New readers should include a minimal synthetic fixture rather than a
large tracked dataset.
