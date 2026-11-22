# Contributing

## Setup

```bash
pip install -e ".[all,dev]"
```

## Style

* **PEP 8**, max line length 99, checked with `ruff check src tests` and
  `flake8 --max-line-length 99 src tests`.
* **numpydoc** docstrings (Parameters / Returns sections) on public functions.
* Type hints on public functions.
* Heavy dependencies (`torch`, `transformers`, `ultralytics`) are imported **inside** the functions or classes that need them, so the core package stays light.
* Every new plot goes in `broilerseg/viz/`, returns the `Figure`, and accepts an optional `ax` where that makes sense.

## Tests

```bash
pytest
```

Tests use `broilerseg.synthetic` (generated videos and a colour segmenter), so they need no GPU and no model downloads. Add a test for every bug you fix.

## Notebooks

Keep notebook logic thin: call package functions and plot. To check that a notebook runs end to end, set `DRY_RUN = True` and run all cells.

## Releasing

Update `__version__` in `src/broilerseg/__init__.py`, `version` in `pyproject.toml`, and `CHANGELOG.md`.
