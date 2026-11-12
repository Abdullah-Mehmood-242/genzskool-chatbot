# ADR-0004: Structure-aware chunking

- Status: Accepted
- Date: 2026-11-12

## Context
Fixed-size chunking would split the course table across pages (TikTok automation sits alone on
page 2) and mix unrelated courses into one chunk, which breaks fee/mentor attribution.

## Decision
Chunk by semantic unit, driven by the PDF's numbered sections and the catalogue table:

| doc_type | One chunk per | Source |
|---|---|---|
| `course` | course (8) | catalogue table row (fee, level, mentor, length) merged with its section 4 detail paragraph |
| `mentor` | mentor (6) | section 5 |
| `policy` | about, how-it-works, enrollment, FAQ/certificate, data limitations | sections 1, 6, 7, 8, 9 |
| `contact` | contact and location | section 2 |
| `overview` | catalogue overview listing all 8 courses with fees | section 3 table |

- Every chunk text starts with a context header, e.g.
  `Course: Shopify store setup to first sale | Mentor: Hamza Ali | Fee: Rs. 30,000`.
- Max ~400 tokens (whitespace-token estimate); longer sections are split on paragraph/bullet
  boundaries with a small overlap (config: `config/chunking.yaml`).
- Course facts are parsed into structured metadata (`fee_pkr`, `level`, `category`, `mentor`,
  `course_slug`). `fee_pkr` is `null` for TikTok automation (fee not published).
- If the structured parse fails (unknown PDF layout), the chunker falls back to section-based
  chunking with `doc_type = policy`, and logs a warning. It never invents metadata.

## Consequences
- (+) Each fee/mentor fact lives in exactly one course chunk plus the overview, so citations
  are precise and filters work.
- (-) The parser is tuned to this PDF's layout. A layout change is caught by the chunking tests
  against the real PDF, which fail loudly (see `scripts/refresh_kb.md`).
