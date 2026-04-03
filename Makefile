.PHONY: reproduce

PYTHON ?=

reproduce:
	@if [ -n "$(PYTHON)" ]; then \
		PYTHON_BIN="$(PYTHON)" bash scripts/reproduce.sh; \
	else \
		bash scripts/reproduce.sh; \
	fi
