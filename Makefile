default: linter tests

install:
	pip install -U pip
	pip install -U -e '.[dev]'

linter:
	flake8 audiocraft && mypy audiocraft
	flake8 tests && mypy tests

tests:
	coverage run -m pytest tests
	coverage report --include 'audiocraft/*'

docs:
	pdoc3 --html -o docs -f audiocraft

dist:
	python setup.py sdist

.PHONY: linter tests docs dist
