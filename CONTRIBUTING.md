# Contributing

## Development setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .[dev,plots]
pre-commit install
```

## Formatting & linting

```bash
ruff check .
ruff format .
```
