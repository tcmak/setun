.PHONY: doctor doctor-lock lint format test sim-smoke resource-report resource-check ci nightly

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
		$(PYTHON) -m targets.sim --with-accel-stub; \
	else \
		echo "[sim-smoke] targets/sim.py not yet implemented (scheduled for Task 0.3)"; \
	fi

resource-report:
	$(PYTHON) scripts/resource_report.py

resource-check:
	$(PYTHON) scripts/resource_report.py --check-baseline

ci:
	@$(PYTHON) scripts/doctor.py --lockfile toolchain.lock --banner
	$(MAKE) lint
	$(MAKE) test
	$(MAKE) resource-check
	$(MAKE) sim-smoke

nightly:
	@$(PYTHON) scripts/doctor.py --lockfile toolchain.lock --banner
	$(MAKE) lint
	$(MAKE) resource-check
	SETUN_NIGHTLY=1 $(PYTHON) -m pytest tests/unit
	$(MAKE) sim-smoke
