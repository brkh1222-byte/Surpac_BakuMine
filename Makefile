UV ?= uv

.PHONY: install check dev e2e

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

# API on :8000 (reload) and the Vite dev server together; Ctrl-C stops both.
dev:
	@trap 'kill $$API $$WEB 2>/dev/null; wait' EXIT; \
	trap 'exit 0' INT TERM; \
	$(UV) run uvicorn bakumine.api:app --port 8000 --reload & API=$$!; \
	(cd web && exec ./node_modules/.bin/vite) & WEB=$$!; \
	wait
