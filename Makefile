# Convenience targets. Everything here is a thin wrapper around a plain
# `python3` command, documented in NOTES.md. Nothing in this repository needs
# `make` -- the commands below are here so the common sequences are one line.

PYTHON ?= python3

.PHONY: help all test generate train evaluate report clean

help:
	@echo "make all       - regenerate dataset, train, evaluate, write report"
	@echo "make test      - run the test suite"
	@echo "make generate  - write data/name_pairs.csv"
	@echo "make train     - write data/model.json"
	@echo "make evaluate  - print the algorithm comparison"
	@echo "make report    - write reports/results.json and reports/results.md"
	@echo "make clean     - remove generated artefacts"

all:
	$(PYTHON) run.py all

test:
	$(PYTHON) run.py test

generate:
	$(PYTHON) -m name_match.cli generate

train:
	$(PYTHON) -m name_match.cli train

evaluate:
	$(PYTHON) -m name_match.cli evaluate

report:
	$(PYTHON) -m name_match.cli report

clean:
	rm -f data/name_pairs.csv data/model.json reports/results.json reports/results.md
	rm -rf name_match/__pycache__ tests/__pycache__
