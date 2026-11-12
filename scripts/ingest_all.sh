#!/usr/bin/env sh
# Trigger ingestion of every PDF in data/pdfs through the running ingestion service and print a report.
# Exits non-zero if any file failed. Usually not needed: the service auto-ingests on startup and on change.
#
#   sh scripts/ingest_all.sh                 # idempotent: unchanged files are skipped
#   FORCE=true sh scripts/ingest_all.sh      # re-embed everything (new index version)
#   INGESTION_URL=http://host:8001 sh scripts/ingest_all.sh
set -eu

URL="${INGESTION_URL:-http://127.0.0.1:8001}"
FORCE="${FORCE:-false}"
# Pick an interpreter that actually runs (on Windows `python3` may be a Microsoft Store stub).
PY=""
for candidate in python3 python py; do
  if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c "import json" >/dev/null 2>&1; then
    PY="$candidate"
    break
  fi
done
[ -n "$PY" ] || { echo "python is required to format the report" >&2; exit 2; }

case "$FORCE" in true | false) ;; *) echo "FORCE must be true or false" >&2; exit 2 ;; esac

response="$(curl -fsS --max-time 300 -X POST "$URL/v1/ingest" \
  -H 'Content-Type: application/json' -d "{\"force\": $FORCE}")" || {
  echo "ingest request to $URL failed (is the stack up? make up)" >&2
  exit 1
}

printf '%s' "$response" | "$PY" -c '
import json, sys
r = json.load(sys.stdin)
for f in r["files"]:
    extra = f["error"] or "; ".join(f["warnings"])
    print("%-16s %-45s chunks=%-4s v=%s %s" % (f["outcome"], f["file"], f["chunks"], f["index_version"], extra))
for doc in r["pruned"]:
    print("%-16s %s" % ("pruned", doc))
print("OK" if r["ok"] else "FAILED")
sys.exit(0 if r["ok"] else 1)
'
