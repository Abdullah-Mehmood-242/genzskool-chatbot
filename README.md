# Genz Skool Career Advisor (RAG)

Grounded career/course advisor for https://genzskool.com.

- `MASTER_PROMPT.md` : the single prompt to give Claude Code to build the system.
- `data/pdfs/`       : source-of-truth knowledge PDFs (one generated from the website, crawled 2026-11-12).
- `.env.example`     : all API keys are placeholders. Copy to `.env` and fill in.
- `services/`        : api_gateway, ingestion, retrieval, advisor, evaluation (each independently deployable).
- `libs/common/`     : shared contracts only (DTOs, errors, logging, config). No business logic.

Start: `claude` in this folder, then paste the contents of MASTER_PROMPT.md.

## Development (Python 3.12)

```bash
py -3.12 -m venv .venv && . .venv/Scripts/activate      # Windows (Git Bash); Linux/macOS: .venv/bin/activate
for f in services/*/requirements.txt; do pip install -r "$f"; done
pip install -r requirements-dev.txt && pre-commit install

python scripts/dev.py check          # lint + mypy --strict + all tests (same as `make check`)
python scripts/dev.py smoke-local    # boot all 5 services with uvicorn (no Docker) and probe them
```

Keep the venv **activated when committing**: the pre-commit hooks call `python scripts/dev.py`.
