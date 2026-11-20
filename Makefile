UV ?= uv
# Port of the API for `make dev` (`make dev API_PORT=8010` when 8000 is taken).
API_PORT ?= 8000

.PHONY: install check dev e2e screenshots

install:
	$(UV) sync
	npm --prefix web install

check:
	$(UV) run ruff check .
	$(UV) run pytest
	npm --prefix web run typecheck
	npm --prefix web run test

# Browser tests of the viewer (Playwright, chromium). Not part of `check`: slower.
e2e:
	npm --prefix web run e2e

# README screenshots (docs/screenshots/): builds a model through the UI on a fresh project.
screenshots:
	npm --prefix web run screenshots

# API on :$(API_PORT) (reload) and the Vite dev server together; Ctrl-C stops both.
dev:
	@trap 'kill $$API $$WEB 2>/dev/null; wait' EXIT; \
	trap 'exit 0' INT TERM; \
	$(UV) run uvicorn bakumine.api:app --port $(API_PORT) --reload & API=$$!; \
	(cd web && BAKUMINE_API_URL=http://127.0.0.1:$(API_PORT) exec ./node_modules/.bin/vite) & WEB=$$!; \
	wait
