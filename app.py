import hashlib
import json
import math
import re
import unicodedata
from datetime import datetime, timezone, timedelta
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

app = FastAPI()

URI_RE = re.compile(r"^gs://[^/\s]+/.+$")
GEN_RE = re.compile(r"^[0-9]+$")
CRC_RE = re.compile(r"^[0-9a-f]{8}$")
TIME_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(\d{1,3}))?(Z|[+-]\d{2}:\d{2})$"
)
ROW_KEYS = {"id", "entity", "eventTime", "revision", "text"}
SAFE_INT_MAX = 9007199254740991
REASON_ORDER = {
    "DUPLICATE": 0,
    "POLICY_INVALID": 1,
    "OUT_OF_WINDOW": 2,
    "TRAIN_CONTAMINATION": 3,
}


def compact(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def utf8(s: str) -> bytes:
    return s.encode("utf-8")


def sorted_reason_codes(codes):
    return sorted(set(codes), key=lambda x: utf8(x))


def canonical_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).lower().strip()
    return re.sub(r"\s+", " ", s, flags=re.UNICODE)


def parse_event_time(value: Any):
    if not isinstance(value, str):
        return None
    m = TIME_RE.fullmatch(value)
    if not m:
        return None

    base, frac, offset = m.groups()
    try:
        dt = datetime.strptime(base, "%Y-%m-%dT%H:%M:%S")
    except ValueError:
        return None

    ms = int((frac or "").ljust(3, "0")) if frac else 0

    if offset == "Z":
        tz = timezone.utc
    else:
        sign = 1 if offset[0] == "+" else -1
        oh, om = map(int, offset[1:].split(":"))
        if oh > 14 or om > 59 or (oh == 14 and om != 0):
            return None
        tz = timezone(sign * timedelta(hours=oh, minutes=om))

    dt = dt.replace(microsecond=ms * 1000, tzinfo=tz).astimezone(timezone.utc)
    return dt


def normalized_event_time(value: str):
    dt = parse_event_time(value)
    if dt is None:
        return None
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def valid_row(row: Any) -> bool:
    if not isinstance(row, dict) or set(row.keys()) != ROW_KEYS:
        return False
    if not all(isinstance(row[k], str) for k in ("id", "entity", "eventTime", "text")):
        return False
    rev = row["revision"]
    if isinstance(rev, bool) or not isinstance(rev, int):
        return False
    if rev < 0 or rev > SAFE_INT_MAX:
        return False
    return parse_event_time(row["eventTime"]) is not None


def crc32c(data: bytes) -> int:
    # Castagnoli CRC32C, reflected polynomial.
    crc = 0xFFFFFFFF
    poly = 0x82F63B78
    for b in data:
        crc ^= b
        for _ in range(8):
            mask = -(crc & 1)
            crc = (crc >> 1) ^ (poly & mask)
    return crc ^ 0xFFFFFFFF


def parse_jsonl(content: str):
    rows = []
    had_json_error = False
    had_schema_error = False
    nonblank = False

    for line in content.splitlines():
        if not line.strip():
            continue
        nonblank = True
        try:
            obj = json.loads(line)
        except Exception:
            had_json_error = True
            continue
        if not valid_row(obj):
            had_schema_error = True
            continue
        rows.append(obj)

    if not nonblank:
        had_schema_error = True

    return rows, had_json_error, had_schema_error


def parse_policy(policy: Any):
    if not isinstance(policy, dict):
        return None
    if not all(k in policy for k in ("minTime", "maxTime", "contaminationThreshold")):
        return None

    min_dt = parse_event_time(policy["minTime"])
    max_dt = parse_event_time(policy["maxTime"])
    threshold = policy["contaminationThreshold"]

    if (
        min_dt is None
        or max_dt is None
        or isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(float(threshold))
        or not (0 <= float(threshold) <= 1)
        or min_dt > max_dt
    ):
        return None

    return min_dt, max_dt, float(threshold)


def word_set(text: str):
    words = []
    current = []
    for ch in text:
        if ch.isalpha() or ch.isnumeric():
            current.append(ch.lower())
        elif current:
            words.append("".join(current))
            current = []
    if current:
        words.append("".join(current))
    return set(words)


def jaccard(a, b):
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union)


def split_for(entity: str) -> str:
    bucket = hashlib.sha256(utf8(entity)).digest()[0] % 10
    if bucket <= 5:
        return "train"
    if bucket <= 7:
        return "validation"
    return "test"


def row_json(row):
    return compact({
        "id": row["id"],
        "entity": row["entity"],
        "eventTime": row["eventTime"],
        "revision": row["revision"],
        "text": row["text"],
    })


def row_sort_key(row):
    return (utf8(row["id"]), utf8(row_json(row)))


def obj_sort_key(obj):
    uri = obj["uri"]
    uri_bytes = utf8(uri) if isinstance(uri, str) else b""
    return (0 if uri is None else 1, uri_bytes, utf8(compact(obj)))


def lineage_sort_key(item):
    return (utf8(item["uri"]), utf8(compact(item)))


def reject_object(uri, reasons):
    return {"uri": uri if isinstance(uri, str) else None,
            "reasonCodes": sorted_reason_codes(reasons)}


def reject_row(row_id, reasons):
    return {"id": row_id, "reasonCodes": sorted_reason_codes(reasons)}


@app.post("/build-corpus")
async def build_corpus(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "INVALID_INPUT"}, status_code=400)

    if not isinstance(body, dict) or "policy" not in body or not isinstance(body.get("objects"), list):
        return JSONResponse({"error": "INVALID_INPUT"}, status_code=400)

    policy = parse_policy(body["policy"])
    objects = body["objects"]

    rejected_objects = []
    valid_objects = []
    all_rows = []

    for supplied in objects:
        # Object shape is not itself assigned a dedicated code; each required
        # field is checked independently where possible.
        if not isinstance(supplied, dict):
            supplied = {}
        uri = supplied.get("uri")
        reasons = []

        if not isinstance(uri, str) or URI_RE.fullmatch(uri) is None:
            reasons.append("URI_INVALID")

        generation = supplied.get("generation")
        fetched_generation = supplied.get("fetchedGeneration")
        gen_valid = isinstance(generation, str) and GEN_RE.fullmatch(generation) is not None
        fetched_valid = isinstance(fetched_generation, str) and GEN_RE.fullmatch(fetched_generation) is not None
        if not gen_valid or not fetched_valid:
            reasons.append("GENERATION_INVALID")
        if gen_valid and fetched_valid and generation != fetched_generation:
            reasons.append("GENERATION_MISMATCH")

        crc = supplied.get("crc32c")
        crc_valid = isinstance(crc, str) and CRC_RE.fullmatch(crc) is not None
        if not crc_valid:
            reasons.append("CRC32C_INVALID")

        content = supplied.get("content")
        if isinstance(content, str) and crc_valid:
            actual = f"{crc32c(utf8(content)):08x}"
            if actual != crc:
                reasons.append("CRC32C_MISMATCH")

        schema_ok = supplied.get("schemaId") == "training-v1" and isinstance(supplied.get("content"), str)
        if not schema_ok:
            reasons.append("SCHEMA_INVALID")

        parsed_rows = []
        jsonl_invalid = False
        schema_invalid = False
        if isinstance(content, str):
            parsed_rows, jsonl_invalid, schema_invalid = parse_jsonl(content)
            if jsonl_invalid:
                reasons.append("JSONL_INVALID")
            if schema_invalid:
                reasons.append("SCHEMA_INVALID")

        # Deduplicate object-level reasons.
        reasons = sorted_reason_codes(reasons)

        if reasons:
            rejected_objects.append(reject_object(uri, reasons))
            continue

        valid_objects.append(supplied)
        for row in parsed_rows:
            all_rows.append({
                "source": supplied,
                "raw": row,
                "id": row["id"],
                "entity": canonical_text(row["entity"]),
                "eventTime": normalized_event_time(row["eventTime"]),
                "revision": row["revision"],
                "text": canonical_text(row["text"]),
            })

    # Deduplicate globally across all object-valid rows.
    winners = {}
    rejected_rows = []
    for item in all_rows:
        key = (item["entity"], item["eventTime"], item["text"])
        candidate = {
            "id": item["id"],
            "entity": item["entity"],
            "eventTime": item["eventTime"],
            "revision": item["revision"],
            "text": item["text"],
        }
        if key not in winners:
            winners[key] = (candidate, item)
            continue

        current, current_item = winners[key]
        better = (
            item["revision"] > current["revision"]
            or (
                item["revision"] == current["revision"]
                and utf8(item["id"]) < utf8(current["id"])
            )
        )
        if better:
            rejected_rows.append(reject_row(current["id"], ["DUPLICATE"]))
            winners[key] = (candidate, item)
        else:
            rejected_rows.append(reject_row(item["id"], ["DUPLICATE"]))

    retained = [pair for pair in winners.values()]

    # Policy applies to every deduplicated retained row.
    train_rows = []
    validation_rows = []
    test_rows = []

    if policy is None:
        for row, _item in retained:
            rejected_rows.append(reject_row(row["id"], ["POLICY_INVALID"]))
    else:
        min_dt, max_dt, threshold = policy
        for row, _item in retained:
            dt = parse_event_time(row["eventTime"])
            if dt < min_dt or dt > max_dt:
                rejected_rows.append(reject_row(row["id"], ["OUT_OF_WINDOW"]))
                continue

            split = split_for(row["entity"])
            if split == "train":
                train_rows.append(row)
            elif split == "validation":
                validation_rows.append(row)
            else:
                test_rows.append(row)

        train_sets = [word_set(r["text"]) for r in train_rows]
        for split_name, rows in (("validation", validation_rows), ("test", test_rows)):
            kept = []
            for row in rows:
                ws = word_set(row["text"])
                contaminated = any(jaccard(ws, tws) >= threshold for tws in train_sets)
                if contaminated:
                    rejected_rows.append(reject_row(row["id"], ["TRAIN_CONTAMINATION"]))
                else:
                    kept.append(row)
            if split_name == "validation":
                validation_rows = kept
            else:
                test_rows = kept

    splits = {
        "train": sorted(train_rows, key=row_sort_key),
        "validation": sorted(validation_rows, key=row_sort_key),
        "test": sorted(test_rows, key=row_sort_key),
    }

    digests = {}
    for name in ("train", "validation", "test"):
        payload = "".join(row_json(r) + "\n" for r in splits[name]).encode("utf-8")
        digests[name] = hashlib.sha256(payload).hexdigest()

    lineage = [
        {
            "uri": obj["uri"],
            "generation": obj["generation"],
            "crc32c": obj["crc32c"],
            "schemaId": obj["schemaId"],
        }
        for obj in valid_objects
    ]
    lineage.sort(key=lineage_sort_key)

    rejected_objects.sort(key=obj_sort_key)
    rejected_rows.sort(key=lambda x: (utf8(x["id"]), utf8(compact(x))))

    result = {
        "splits": splits,
        "rejectedObjects": rejected_objects,
        "rejectedRows": rejected_rows,
        "digests": digests,
        "lineage": lineage,
    }

    return Response(
        content=compact(result),
        media_type="application/json",
    )
