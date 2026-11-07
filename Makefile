PYTHON ?= python3
COMPOSE ?= docker compose

ifneq (,$(wildcard .env))
include .env
export
endif

.PHONY: validate test manifest preflight build up down logs ps smoke bench

# GPU-free checks (no Docker needed)
validate:
	$(PYTHON) scripts/validate_repo.py

test:
	$(PYTHON) -B -m unittest discover -s tests -p 'test_*.py'

manifest:
	$(PYTHON) scripts/update_manifest.py

# Server
preflight:
	./scripts/preflight.sh

build: validate
	$(COMPOSE) build

up: preflight
	$(COMPOSE) up -d

down:
	$(COMPOSE) down

logs:
	$(COMPOSE) logs -f --tail=200

ps:
	$(COMPOSE) ps

smoke:
	$(PYTHON) scripts/smoke_test.py --base-url http://127.0.0.1:$(or $(PORT),8000)

bench:
	$(PYTHON) scripts/bench.py --base-url http://127.0.0.1:$(or $(PORT),8000) --out results/bench-$(shell date +%Y%m%d-%H%M%S).json
