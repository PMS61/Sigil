# AGENTS.md - Sigil Gesture Control Framework

## Build & Development Commands

```bash
# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Run linter (ruff)
ruff check sigil/

# Run type checker (mypy)
mypy sigil/

# Run all tests
pytest tests/

# Run a single test file
pytest tests/test_classifier.py -v

# Run a single test function
pytest tests/test_classifier.py::test_continuous_gesture_bypasses_blanking -v

# Run with coverage
pytest tests/ --cov=sigil --cov-report=term-missing
```

## Code Style Guidelines

### Imports
- Use `from __future__ import annotations` at top of every module
- Group imports: stdlib → third-party → local (separated by blank lines)
- Use explicit imports, avoid `import *`
- Lazy imports for optional heavy dependencies (mediapipe, GTK)

### Formatting
- Line length: 100 characters (configured in ruff)
- Use double quotes for strings
- Trailing commas in multi-line collections
- No trailing whitespace

### Type Hints
- All function signatures must have return types
- Use `X | None` syntax (Python 3.10+ style via future annotations)
- Use `list[X]`, `dict[K, V]` (lowercase generics)
- Private attributes: prefix with `_`
- Use `Any` sparingly, prefer specific types

### Naming
- Classes: `PascalCase`
- Functions/methods: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Private: `_leading_underscore`
- Module-level logger: `logger = logging.getLogger(__name__)`

### Documentation
- Module docstrings: Brief description + section reference (§5.2)
- Class docstrings: One-line purpose
- Method docstrings: Google style, explain parameters/returns if non-obvious
- Use `# ── Section ────────────────────` for visual separation

### Dataclasses
- Use `@dataclass` for data containers
- Use `field(default_factory=...)` for mutable defaults
- Order: required fields first, then optional with defaults

### Error Handling
- Use specific exception types
- Log errors with `logger.exception()` for tracebacks
- Don't suppress exceptions silently
- Use `try/except` for optional feature detection (e.g., MediaPipe APIs)

### Testing
- Use pytest with `unittest.mock.patch` for mocking
- Test files: `tests/test_<module>.py`
- Test functions: `test_<description>`
- Mock external dependencies (camera, MediaPipe, hyprctl)
- Use `FrameResult` and `HandResult` from `sigil.tracker` for test data

### Config
- YAML config at `~/.config/sigil/config.yaml`
- Gestures defined in `sigil/default_config.yaml`
- Use `load_config()` / `save_config()` from `sigil.config`
- Config changes hot-reloadable via `daemon.reload_config()`
