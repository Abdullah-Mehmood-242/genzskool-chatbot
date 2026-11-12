# Refreshing the knowledge base (manual, documented)

The advisor never scrapes genzskool.com at runtime. `data/pdfs/` is the only source of truth.
**Stale fees and batch dates are the top operational risk**: refresh the PDF whenever the website
changes (new course, fee change, new batch) and at least monthly.

## 1. Regenerate the PDF
1. Crawl the public pages: homepage, `/courses`, every `/courses/<slug>`, `/mentors`,
   `/how-it-works`, `/contact`, `/enroll`.
2. Produce a PDF with the **same structure** as the current one, because the chunker relies on it
   (ADR-0004):
   - numbered top-level sections `1. About ...` to `9. Known data limitations ...`, in order;
   - section 3 holds the catalogue **as a table** with the header
     `Course | Category | Lessons / Length | Level | Mentor | Fee` (columns may move; names matter);
   - section 4 has one block per course: title line, then `Page: https://genzskool.com/courses/<slug>`;
   - section 5 lists mentors as `- Name - Role - teaches X, Y.`;
   - keep the crawl date on page 1 and keep section 9 (data limitations) up to date.
3. Never add facts that are not on the website (no salaries, placement rates or invented syllabi).

## 2. Validate before it goes live
```bash
cp new_kb.pdf data/pdfs/genzskool_knowledge_base.pdf     # or add a second PDF next to it
python scripts/dev.py test ingestion                       # the real-PDF chunking tests pin every fact
```
`tests/domain/test_chunking_real_pdf.py` asserts the exact courses, fees, mentors and policies. When
the website really changed, those expectations must be updated in the same commit, which makes the
fee/mentor change visible in code review. If the tests fail because of **layout** (e.g. the
catalogue table is missing), fix the PDF rather than the chunker.

## 3. Publish
- With the stack running nothing else is needed: **auto-ingest** (ADR-0011) notices the changed
  file within `AUTO_INGEST_INTERVAL_S` seconds, builds a new index version, switches to it, and
  deletes the old one. Check with `GET :8001/v1/ingest/status` (or `make ingest` for a report).
- If the new PDF fails to parse, the previous index version keeps serving and the status shows
  `last_ok: false` with the reason.
- Removing a PDF from `data/pdfs/` removes it from the index (pruning), unless the folder is empty.

## 4. After publishing
Run the evaluation gate (`make eval`) and spot-check a few fees through the chat API.
