.PHONY: doctor doctor-lock lint format test sim-smoke ci nightly

PYTHON ?= python3

doctor:
	@$(PYTHON) scripts/doctor.py --lockfile toolchain.lock

doctor-lock:
	@$(PYTHON) scripts/doctor.py --lockfile toolchain.lock --write

lint:
	$(PYTHON) -m ruff check setun targets tests scripts
	$(PYTHON) -m ruff format --check setun targets tests scripts

format:
	$(PYTHON) -m ruff check --fix setun targets tests scripts
	$(PYTHON) -m ruff format setun targets tests scripts

test:
	$(PYTHON) -m pytest tests/unit

sim-smoke:
	@if [ -f targets/sim.py ]; then \
		$(PYTHON) -m targets.sim; \
	else \
		echo "[sim-smoke] targets/sim.py not yet implemented (scheduled for Task 0.3)"; \
	fi

ci:
	@$(PYTHON) scripts/doctor.py --lockfile toolchain.lock --banner
	$(MAKE) lint
	$(MAKE) test
	$(MAKE) sim-smoke

nightly:
	@$(PYTHON) scripts/doctor.py --lockfile toolchain.lock --banner
	pytest tests/
