INTEG=AUDIOCRAFT_DORA_DIR="/tmp/magma_$(USER)" python3 -m dora -v run --clear device=cpu dataset.num_workers=0 optim.epochs=1 \
	dataset.train.num_samples=10 dataset.valid.num_samples=10 \
	dataset.evaluate.num_samples=10 dataset.generate.num_samples=2 sample_rate=16000 \
	logging.level=DEBUG
INTEG_COMPRESSION = $(INTEG) solver=compression/debug rvq.n_q=2 rvq.bins=48 checkpoint.save_last=true   # SIG is 5091833e
INTEG_MUSICGEN = $(INTEG) solver=musicgen/debug dset=audio/example compression_model_checkpoint=//sig/5091833e \
	transformer_lm.n_q=2 transformer_lm.card=48 transformer_lm.dim=16 checkpoint.save_last=false  # Using compression model from 5091833e
INTEG_AUDIOGEN = $(INTEG) solver=audiogen/debug dset=audio/example compression_model_checkpoint=//sig/5091833e \
	transformer_lm.n_q=2 transformer_lm.card=48 transformer_lm.dim=16 checkpoint.save_last=false  # Using compression model from 5091833e
INTEG_MBD = $(INTEG) solver=diffusion/debug dset=audio/example  \
	checkpoint.save_last=false  # Using compression model from 616d7b3c

PYTHON_VENV_PATH:=venv

default: linter tests

install: ## Install dependencies
	@pip install -U pip
	@pip install -U -e '.[dev]'

linter: ## Run linter
	@flake8 audiocraft && mypy audiocraft
	@flake8 tests && mypy tests

tests: ## Run unit tests
	@coverage run -m pytest tests
	@coverage report

tests_integ: ## Run integration tests
	$(INTEG_COMPRESSION)
	$(INTEG_MBD)
	$(INTEG_MUSICGEN)
	$(INTEG_AUDIOGEN)

api_docs: ## Generate API documentation
	@pdoc3 --html -o api_docs -f audiocraft

dist: ## Build source distribution
	@python setup.py sdist

help: ## Help
	@grep --no-filename -E '^[a-zA-Z_/-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

venv: ## Create python virtual environment
	@python3 -m venv $(PYTHON_VENV_PATH)
	@export PATH="$(PYTHON_VENV_PATH)/bin:$$PATH" && \
		pip3 install pip --upgrade --index-url=https://pypi.org/simple/
	@source $(PYTHON_VENV_PATH)/bin/activate || true
	@$(PYTHON_VENV_PATH)/bin/pip3 install -r requirements.txt

.PHONY: linter tests api_docs dist venv help
