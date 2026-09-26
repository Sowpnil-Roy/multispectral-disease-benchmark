# Reproduce the paper end to end:  make reproduce
PY ?= python
export OMP_NUM_THREADS = 1
export OPENBLAS_NUM_THREADS = 1
export MKL_NUM_THREADS = 1

.PHONY: help data stages figures tables verify test reproduce compare clean-results

help:
	@echo "make verify     check every number in the paper against results/ (seconds)"
	@echo "make test       unit tests + paper-claim check"
	@echo "make reproduce  regenerate data, rerun all stages, figures, tables, verify (~12 min)"
	@echo "make compare    compare a reproduction (results/) with the paper run (results_ref/)"

data/spectra.npz:
	$(PY) src/simulate.py

data: data/spectra.npz

stages: data
	$(PY) src/stages.py all

figures:
	$(PY) src/figs.py

tables:
	$(PY) src/tables.py

verify:
	$(PY) src/verify_paper.py

test:
	$(PY) -m pytest -q

clean-results:
	mkdir -p results_ref && cp results/*.json results_ref/
	rm -f results/*.json

compare:
	$(PY) src/compare_results.py results_ref results

reproduce: clean-results data stages figures tables compare verify
