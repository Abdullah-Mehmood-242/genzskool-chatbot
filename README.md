# Genz Skool Career Advisor (RAG)

Grounded career/course advisor for https://genzskool.com.

- `MASTER_PROMPT.md` : the single prompt to give Claude Code to build the system.
- `data/pdfs/`       : source-of-truth knowledge PDFs (one generated from the website, crawled 2026-11-12).
- `.env.example`     : all API keys are placeholders. Copy to `.env` and fill in.
- `services/`        : api_gateway, ingestion, retrieval, advisor, evaluation (each independently deployable).
- `libs/common/`     : shared contracts only (DTOs, errors, logging, config). No business logic.

Start: `claude` in this folder, then paste the contents of MASTER_PROMPT.md.
