# AcmeCloud evals — common tasks. Override PY to use another interpreter.
VENV ?= .venv
PY   ?= $(VENV)/bin/python
WORKERS ?= 4

.PHONY: setup doctor test eval eval-live compare baseline models calibrate clean

STAMP = $(VENV)/.installed

$(STAMP): requirements.txt
	python3 -m venv $(VENV)
	$(PY) -m pip install -q --upgrade pip
	$(PY) -m pip install -q -r requirements.txt
	@touch $(STAMP)

setup: $(STAMP)   ## create venv and install deps
	@test -f .env || (cp .env.example .env && echo "Created .env from .env.example (mock mode).")

doctor: $(STAMP)           ## check env + one tiny live call per model (never prints secrets)
	$(PY) -m app.doctor

test: $(STAMP)             ## unit tests for app + graders (no network)
	EVAL_MODE=mock $(PY) -m pytest

eval: $(STAMP)             ## full eval suite in deterministic mock mode
	$(PY) -m evals.runner --mode mock --workers $(WORKERS)

eval-live: $(STAMP)        ## full eval suite against real models via LiteLLM (uses .env)
	$(PY) -m evals.runner --mode live --workers $(WORKERS)

compare: $(STAMP)          ## diff results/latest.json against results/baseline.json
	$(PY) -m evals.compare

baseline:         ## promote the latest run to the committed baseline
	cp results/latest.json results/baseline.json
	@echo "Baseline updated. Review and commit results/baseline.json."

models: $(STAMP)           ## compare several models: make models MODELS=openai/a,openai/b
	$(PY) -m evals.runner --mode live --models "$(MODELS)" --workers $(WORKERS)

calibrate: $(STAMP)        ## judge-vs-human agreement for the configured JUDGE_MODEL
	$(PY) -m evals.judge_calibration.calibrate

clean:
	rm -rf .pytest_cache results/*.md results/runs
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
