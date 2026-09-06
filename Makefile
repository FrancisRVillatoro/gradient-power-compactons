PYTHON ?= python3

.PHONY: quick figures generator static clean

quick:
	$(PYTHON) scripts/check_fast.py

figures:
	$(PYTHON) scripts/generate_figures.py

generator:
	$(PYTHON) scripts/generator_galerkin.py

static:
	cd data && $(PYTHON) ../scripts/numerics_static.py

clean:
	rm -rf __pycache__ scripts/__pycache__ scratch
