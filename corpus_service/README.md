# Deterministic Corpus + Stateful BQML Gate

FastAPI service exposing:

- `POST /build-corpus` — deterministic JSONL corpus validation, canonicalization, deduplication, splitting, contamination filtering, digests, and lineage.
- `POST /bqml` — stateful two-phase experiment gate. `phase=select` freezes a successful selection under `runId`; `phase=evaluate` validates the frozen lineage and applies final-test metric, slice, and byte gates.

## Local

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

## Render

Build command:

```bash
pip install -r requirements.txt
```

Start command:

```bash
uvicorn app:app --host 0.0.0.0 --port $PORT
```

The public base URL is the Render service URL. The grader appends `/build-corpus` or `/bqml`.

BQML selection state is held in the service process, so replays and conflicts work while the same service instance is running. No external database is required.
