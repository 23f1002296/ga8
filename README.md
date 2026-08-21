# Deterministic JSONL Corpus Service

FastAPI service implementing `POST /build-corpus`.

## Local run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

The endpoint is:

```text
POST http://127.0.0.1:8000/build-corpus
```

## Render

Create a Render Web Service from this repository.

- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`

After deployment, the public base URL is the service URL, for example:

`https://<your-service-name>.onrender.com`

Submit that base URL, without `/build-corpus`.
