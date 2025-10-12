# AudioCraft UV Migration Guide

This document outlines the migration of AudioCraft from traditional pip/setuptools to UV for dependency management.

## What Changed

### 1. New Configuration Files

- **`pyproject.toml`**: Complete project configuration with dependencies, build system, and tool configurations
- **`.python-version`**: Pinned Python version (3.9) for consistent development
- **`Makefile.uv`**: UV-based Makefile with common development tasks
- **Updated `.gitignore`**: Added UV-specific entries (`uv.lock`, `.python-version`)

### 2. Updated Files

- **`setup.py`**: Simplified to be compatible with pyproject.toml (kept for backward compatibility)
- **`README.md`**: Added UV installation instructions and development workflow
- **`requirements.txt`**: Still present but dependencies are now managed in pyproject.toml

### 3. Dependency Management

All dependencies have been migrated from `requirements.txt` to `pyproject.toml` with:
- Proper version constraints
- Optional dependency groups (dev, wm, all)
- Build dependencies for packages that need them
- Compatible versions for PyTorch ecosystem

## Installation

### Quick Start

```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
git clone https://github.com/facebookresearch/audiocraft.git
cd audiocraft

# Install dependencies
uv sync                    # Basic installation
uv sync --extra dev        # With development tools
uv sync --extra wm         # With watermarking support
uv sync --extra all        # Everything
```

### Using Makefile

```bash
# Development setup
make -f Makefile.uv dev-setup

# Common tasks
make -f Makefile.uv install     # Install dependencies
make -f Makefile.uv format      # Format code
make -f Makefile.uv lint        # Run linting
make -f Makefile.uv test        # Run tests
make -f Makefile.uv check       # Run all checks
```

## Running Commands

### Script Execution

```bash
# Run Python scripts
uv run python your_script.py

# Run training
uv run audiocraft/train.py solvers/musicgen/musicgen_base_32khz

# Run demos
uv run python demos/musicgen_demo.ipynb
```

### Development Tasks

```bash
# Code formatting
uv run black audiocraft tests
uv run isort audiocraft tests

# Linting
uv run flake8 audiocraft tests
uv run mypy audiocraft

# Testing
uv run pytest
uv run pytest --cov=audiocraft
```

## Dependency Management

### Adding Dependencies

```bash
# Runtime dependency
uv add requests

# Development dependency
uv add --dev pytest-cov

# Optional dependency
uv add --optional watermarking audioseal

# Remove dependency
uv remove requests
```

### Updating Dependencies

```bash
# Update lock file
uv lock --upgrade

# Sync with updated dependencies
uv sync
```

## Key Benefits

1. **Speed**: UV is 10-100x faster than pip
2. **Reliability**: Better dependency resolution and conflict detection
3. **Consistency**: Lock file ensures reproducible builds
4. **Simplicity**: Single tool for all Python package management
5. **Modern**: Uses pyproject.toml standard for configuration

## Compatibility

- **Python**: Requires Python 3.9+ (upgraded from 3.8+)
- **PyTorch**: Maintained compatibility with PyTorch 2.1.0 ecosystem
- **Transformers**: Constrained to compatible versions (<4.44.0)
- **Build Dependencies**: Properly configured for packages requiring compilation

## Troubleshooting

### Build Issues

If you encounter build issues:

```bash
# Install with build isolation disabled
uv sync --no-build-isolation

# Install build dependencies manually
uv pip install Cython
```

### Version Conflicts

```bash
# Check current versions
uv pip list

# Reinstall with clean environment
rm -rf .venv
uv sync
```

## Migration Notes

- The original `requirements.txt` is preserved for reference
- `setup.py` is simplified but kept for backward compatibility
- All development tools are now configured in `pyproject.toml`
- UV lock file (`uv.lock`) should be committed for reproducible builds

## Next Steps

1. Test the installation on different platforms
2. Update CI/CD pipelines to use UV
3. Consider removing legacy files once migration is complete
4. Train team members on UV workflow
