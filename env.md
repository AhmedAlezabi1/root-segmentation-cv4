# CV4 — Development Environment

Development environment for the CV4 plant root segmentation project. This repository uses [uv](https://docs.astral.sh/uv/) for package management and virtual environment handling.

## Prerequisites

- **Python 3.12** — check with `python --version`
- **Git** — check with `git --version`
- **uv** — Python package manager (installation below)

## Setup

### 1. Install uv

```bash
pip install uv
```

### 2. Clone and install

```bash
git clone <repo-url>
cd CV4
uv sync
```

This creates a virtual environment and installs all dependencies automatically. The `uv.lock` file ensures everyone gets the exact same versions.

### 3. Install pre-commit hooks

```bash
uv run pre-commit install
```

> **All team members must run this command once after cloning the repo.** This enables automatic code quality checks (linting and formatting) before every commit. If you skip this step, your code won't be checked locally before committing.

### 4. Verify installation

Run the linter to check everything is set up:

```bash
uv run ruff check .
```

You should see `All checks passed!`

Test that key libraries are working:

```bash
uv run python -c "import cv2; print('OpenCV:', cv2.__version__)"
uv run python -c "import numpy; print('NumPy:', numpy.__version__)"
uv run python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"
```

## Usage

Always prefix commands with `uv run` to use the project environment:

```bash
uv run python your_script.py      # run a script
uv run ruff check .               # lint your code
uv run ruff format .              # format your code
uv run pytest                     # run tests
```

## Pre-commit Hooks

The project uses [ruff](https://docs.astral.sh/ruff/) via pre-commit hooks to automatically check and format code before every commit.

**What happens when you commit:**

1. `ruff --fix` runs — catches and auto-fixes linting issues (unused imports, bad patterns, import order)
2. `ruff-format` runs — auto-formats your code (spacing, quotes, line length)
3. If issues are found and auto-fixed, the commit is **rejected** — you need to `git add .` the fixes and commit again
4. If everything passes, the commit goes through

**What the hooks check:**

| Check | Caught? | Auto-fixed? |
|-------|---------|-------------|
| Unused imports | Yes | Yes |
| Bad spacing / formatting | Yes | Yes |
| Trailing whitespace | Yes | Yes |
| Import order | Yes | Yes |
| Lines too long (>120 chars) | Yes | Yes |

**What the hooks do NOT check:**

- Missing docstrings
- Type errors (would need mypy)
- Logic bugs (need unit tests for that)

**If a hook fails**, read the error message, run `git add .` to stage the auto-fixes, and commit again.

## Managing Dependencies

```bash
uv add package_name           # add a runtime dependency
uv add --dev package_name     # add a dev dependency
uv remove package_name        # remove a dependency
```

After any change, commit the updated `pyproject.toml` and `uv.lock` so others can run `uv sync` to stay in sync.

## Dependencies

> These are the initial dependencies. More will be added as the project develops.

**Runtime:** TensorFlow, OpenCV, scikit-image, NumPy, SciPy, Pandas, NetworkX, Matplotlib, FastAPI, Uvicorn, python-multipart

**Dev tools:** Ruff (linting and formatting), Pytest (testing), Pre-commit (git hooks), Sphinx (documentation)

See `pyproject.toml` for exact versions.

## Platform Compatibility

TensorFlow is installed automatically based on your operating system:

- **Windows** — `tensorflow-intel 2.18.0`
- **macOS / Linux** — `tensorflow 2.18.0`

No manual setup needed. This is handled in `pyproject.toml` via platform markers.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `uv` not found | Run `pip install uv` first |
| `uv sync` fails | Make sure Python 3.12 is installed (`python --version`) |
| Module not found | Use `uv run python ...` not just `python ...` |
| TensorFlow import error | Check that your system meets TensorFlow's requirements |
| Pre-commit hooks not running | Run `uv run pre-commit install` |
| Commit rejected by hooks | Run `git add .` to stage auto-fixes, then commit again |
| Ruff complains about formatting | Run `uv run ruff format .` to auto-fix |