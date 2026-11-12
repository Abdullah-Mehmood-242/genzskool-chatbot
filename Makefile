# Thin wrapper. All logic lives in scripts/dev.py so it also runs on Windows without make:
#   python scripts/dev.py <command>

PY ?= python
COMPOSE ?= docker compose

.PHONY: help up down logs ps build test test-common lint fmt typecheck check ingest eval smoke

help:
	@echo "make up | down | logs | build | test | lint | fmt | typecheck | check | ingest | eval | smoke"

up:            ## build and start qdrant + 5 services, wait until healthy
	$(COMPOSE) up -d --build --wait

down:
	$(COMPOSE) down

logs:
	$(COMPOSE) logs -f --tail=100

ps:
	$(COMPOSE) ps

build:
	$(COMPOSE) build

test:          ## unit + contract + architecture tests, every service in its own process
	$(PY) scripts/dev.py test

test-common:
	$(PY) scripts/dev.py test common

lint:
	$(PY) scripts/dev.py lint

fmt:
	$(PY) scripts/dev.py fmt

typecheck:
	$(PY) scripts/dev.py typecheck

check: lint typecheck test

ingest:        ## index data/pdfs via the running ingestion service
	sh scripts/ingest_all.sh

eval:          ## golden-set evaluation; non-zero exit if a threshold is missed
	$(PY) scripts/dev.py eval

smoke:
	sh scripts/smoke_test.sh
