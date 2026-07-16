PYTHON ?= python3.11
VENV := venv
REQ := requirements.txt
CONFIG := configs/config.yml

# venv layout differs between Windows (Scripts/) and POSIX (bin/)
ifeq ($(OS),Windows_NT)
    VENV_BIN := $(VENV)/Scripts
else
    VENV_BIN := $(VENV)/bin
endif

.PHONY: all
all: setup run

.PHONY: setup
setup:
	$(PYTHON) -m venv $(VENV)
	$(VENV_BIN)/pip install -r $(REQ)

.PHONY: run
run:
	$(VENV_BIN)/python main.py --config $(CONFIG)

.PHONY: test
test:
	$(VENV_BIN)/pytest tests/

.PHONY: run-analysis
run-analysis:
	$(VENV_BIN)/python color_code_experiments/run_analysis.py

.PHONY: run-comparison
run-comparison:
	$(VENV_BIN)/python run_comparison.py

.PHONY: clean
clean:
	rm -f process_log*.txt
	rm -rf logs/ .pytest_cache/ .hypothesis/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .ipynb_checkpoints -exec rm -rf {} +

.PHONY: reset
reset: clean
	rm -rf $(VENV)
