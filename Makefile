.PHONY: doctor doctor-lock lint format test coverage coverage-full sim-smoke sim-m0 resource-report resource-check ci nightly

PYTHON ?= python3
PYTEST_PARALLEL ?= -n auto --dist loadfile

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
	$(PYTHON) -m pytest $(PYTEST_PARALLEL) tests/unit

coverage:
	$(PYTHON) -m scripts.coverage_report --fast --show-missing

coverage-full:
	$(PYTHON) -m scripts.coverage_report --show-missing

sim-smoke:
	@if [ -f targets/sim.py ]; then \
		$(PYTHON) -m targets.sim --with-accel-stub; \
	else \
		echo "[sim-smoke] targets/sim.py not yet implemented (scheduled for Task 0.3)"; \
	fi

resource-report:
	$(PYTHON) scripts/resource_report.py

resource-check:
	$(PYTHON) scripts/resource_report.py --check

ci:
	@$(PYTHON) scripts/doctor.py --lockfile toolchain.lock --banner
	$(MAKE) lint
	$(MAKE) test
	$(MAKE) resource-check
	$(MAKE) sim-smoke

sim-m0:
	$(PYTHON) -m scripts.sim_m0

nightly:
	@$(PYTHON) scripts/doctor.py --lockfile toolchain.lock --banner
	$(MAKE) lint
	$(MAKE) resource-check
	SETUN_NIGHTLY=1 $(PYTHON) -m pytest $(PYTEST_PARALLEL) tests/unit
	$(MAKE) sim-m0
	$(MAKE) sim-smoke
