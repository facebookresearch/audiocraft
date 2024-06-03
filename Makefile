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
INTEG_WATERMARK = AUDIOCRAFT_DORA_DIR="/tmp/wm_$(USER)" dora run device=cpu dataset.num_workers=0 optim.epochs=1 \
    dataset.train.num_samples=10 dataset.valid.num_samples=10 dataset.evaluate.num_samples=10 dataset.generate.num_samples=10 \
	logging.level=DEBUG solver=watermark/robustness checkpoint.save_last=false dset=audio/example 

default: linter tests

install:
	pip install -U pip
	pip install -U -e '.[dev]'

linter:
	flake8 audiocraft && mypy audiocraft
	flake8 tests && mypy tests

tests:
	coverage run -m pytest tests
	coverage report

tests_integ:
	$(INTEG_COMPRESSION)
	$(INTEG_MBD)
	$(INTEG_MUSICGEN)
	$(INTEG_AUDIOGEN)
	$(INTEG_WATERMARK)


api_docs:
	pdoc3 --html -o api_docs -f audiocraft

dist:
	python setup.py sdist

.PHONY: linter tests api_docs dist
