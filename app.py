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


# ---------------------------------------------------------------------------
# Stateful two-phase BQML experiment gate
# ---------------------------------------------------------------------------

BQML_RUNS: dict[str, dict[str, Any]] = {}
BQML_LOCK = __import__("threading").Lock()


def bqml_json(obj):
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def bqml_error(status_code: int, code: str):
    return Response(
        content=bqml_json({"error": code}),
        status_code=status_code,
        media_type="application/json",
    )


def bqml_utf8(s):
    return s.encode("utf-8")


def bqml_codes(codes):
    return sorted(set(codes), key=bqml_utf8)


def bqml_valid_safe_int(value, positive=False):
    if isinstance(value, bool) or not isinstance(value, int):
        return False
    if positive:
        return 1 <= value <= SAFE_INT_MAX
    return 0 <= value <= SAFE_INT_MAX


def bqml_valid_run_id(value):
    return isinstance(value, str) and 0 < len(value) <= 128


def bqml_time(value):
    return parse_event_time(value)


def bqml_valid_selection_row(row):
    if not isinstance(row, dict):
        return False
    required = {
        "id", "entity", "eventTime", "predictionTime",
        "version", "split", "features"
    }
    if set(row.keys()) != required:
        return False
    if not isinstance(row["id"], str) or not row["id"]:
        return False
    if not isinstance(row["entity"], str) or not row["entity"]:
        return False

    event_dt = bqml_time(row["eventTime"])
    prediction_dt = bqml_time(row["predictionTime"])
    if event_dt is None or prediction_dt is None:
        return False
    if event_dt > prediction_dt:
        return False

    if not bqml_valid_safe_int(row["version"]):
        return False
    if row["split"] not in ("TRAIN", "EVAL"):
        return False
    if not isinstance(row["features"], dict):
        return False

    for name, feature in row["features"].items():
        if not isinstance(name, str) or not name or not isinstance(feature, dict):
            return False
        if "value" not in feature or "availableAt" not in feature:
            return False
        available_at = bqml_time(feature["availableAt"])
        if available_at is None:
            return False
        if available_at > prediction_dt:
            return False

    return True


def bqml_valid_trial(trial):
    if not isinstance(trial, dict):
        return False
    if set(trial.keys()) != {"trialId", "status", "evalMetric"}:
        return False
    if not bqml_valid_safe_int(trial["trialId"]):
        return False
    if trial["status"] not in ("SUCCEEDED", "FAILED"):
        return False

    # The contract only requires finite evalMetric for SUCCEEDED trials.
    # FAILED trials are never eligible.
    if trial["status"] == "SUCCEEDED":
        metric = trial["evalMetric"]
        if (
            isinstance(metric, bool)
            or not isinstance(metric, (int, float))
            or not math.isfinite(float(metric))
        ):
            return False

    return True


def bqml_selection_fingerprint(body):
    # Canonical object-key ordering makes semantically identical JSON requests
    # replay the same stored response regardless of incoming key order.
    canonical = json.dumps(
        body,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def bqml_digest(train_ids, eval_ids, feature_names):
    payload = bqml_json({
        "trainRowIds": train_ids,
        "evalRowIds": eval_ids,
        "featureNames": feature_names,
    }).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def bqml_validate_selection(body):
    reasons = []

    if not isinstance(body, dict):
        return ["INVALID_INPUT"]

    if body.get("phase") != "select":
        reasons.append("INVALID_INPUT")

    if not bqml_valid_run_id(body.get("runId")):
        reasons.append("INVALID_INPUT")

    forbidden = body.get("forbiddenFeatures")
    if not isinstance(forbidden, list):
        reasons.append("INVALID_INPUT")
    elif any(not isinstance(x, str) for x in forbidden):
        reasons.append("INVALID_INPUT")

    limit = body.get("numTrialsLimit")
    if not bqml_valid_safe_int(limit, positive=True):
        reasons.append("INVALID_INPUT")

    rows = body.get("rows")
    if not isinstance(rows, list) or not rows:
        reasons.append("INVALID_INPUT")
    else:
        if any(not bqml_valid_selection_row(row) for row in rows):
            reasons.append("INVALID_INPUT")

        ids = [row.get("id") for row in rows if isinstance(row, dict)]
        if len(ids) != len(set(ids)):
            reasons.append("INVALID_INPUT")

    trials = body.get("trials")
    if not isinstance(trials, list):
        reasons.append("INVALID_INPUT")
    else:
        if any(not bqml_valid_trial(trial) for trial in trials):
            reasons.append("INVALID_INPUT")

        ids = [trial.get("trialId") for trial in trials if isinstance(trial, dict)]
        if len(ids) != len(set(ids)):
            reasons.append("INVALID_INPUT")

        if bqml_valid_safe_int(limit, positive=True) and len(trials) > limit:
            reasons.append("TRIAL_LIMIT_EXCEEDED")

    return bqml_codes(reasons)


def bqml_deduplicate_rows(rows):
    """
    Deduplicate by [entity, UTC(eventTime)].

    Highest version wins. Exact version ties are broken by UTF-8-smallest ID.
    """
    groups = {}

    for row in rows:
        event_dt = bqml_time(row["eventTime"])
        prediction_dt = bqml_time(row["predictionTime"])

        normalized = dict(row)
        normalized["_event_dt"] = event_dt
        normalized["_prediction_dt"] = prediction_dt
        normalized["_event_utc"] = normalized_event_time(row["eventTime"])
        normalized["_prediction_utc"] = normalized_event_time(row["predictionTime"])

        key = (row["entity"], normalized["_event_utc"])
        groups.setdefault(key, []).append(normalized)

    retained = []

    for group in groups.values():
        winner = sorted(
            group,
            key=lambda row: (
                -row["version"],
                bqml_utf8(row["id"]),
            ),
        )[0]
        retained.append(winner)

    return retained


def bqml_shared_features(retained, forbidden):
    if not retained:
        return []

    common = set(retained[0]["features"].keys())
    for row in retained[1:]:
        common.intersection_update(row["features"].keys())

    forbidden_set = set(forbidden)
    eligible = []

    for name in common:
        if name in forbidden_set:
            continue

        usable = True
        for row in retained:
            available_at = bqml_time(
                row["features"][name]["availableAt"]
            )
            if available_at > row["_prediction_dt"]:
                usable = False
                break

        if usable:
            eligible.append(name)

    return sorted(eligible, key=bqml_utf8)


def bqml_build_selection(body):
    reasons = bqml_validate_selection(body)

    run_id = body.get("runId") if isinstance(body, dict) else ""

    result = {
        "runId": run_id if isinstance(run_id, str) else "",
        "selectedTrialId": None,
        "trainRowIds": [],
        "evalRowIds": [],
        "featureNames": [],
        "datasetDigest": None,
        "reasonCodes": reasons,
    }

    if "INVALID_INPUT" in reasons:
        result["datasetDigest"] = None
        return result

    retained = bqml_deduplicate_rows(body["rows"])

    train_ids = sorted(
        [r["id"] for r in retained if r["split"] == "TRAIN"],
        key=bqml_utf8,
    )
    eval_ids = sorted(
        [r["id"] for r in retained if r["split"] == "EVAL"],
        key=bqml_utf8,
    )
    feature_names = bqml_shared_features(
        retained,
        body["forbiddenFeatures"],
    )

    result["trainRowIds"] = train_ids
    result["evalRowIds"] = eval_ids
    result["featureNames"] = feature_names
    result["datasetDigest"] = bqml_digest(
        train_ids,
        eval_ids,
        feature_names,
    )

    if "TRIAL_LIMIT_EXCEEDED" in reasons:
        result["reasonCodes"] = ["TRIAL_LIMIT_EXCEEDED"]
        return result

    successful = [
        trial
        for trial in body["trials"]
        if trial["status"] == "SUCCEEDED"
        and math.isfinite(float(trial["evalMetric"]))
    ]

    if not successful:
        result["selectedTrialId"] = None
        result["reasonCodes"] = ["NO_SUCCESSFUL_TRIAL"]
        return result

    # Highest metric first. Exact ties use smallest integer trialId.
    best = min(
        successful,
        key=lambda trial: (
            -float(trial["evalMetric"]),
            trial["trialId"],
        ),
    )

    result["selectedTrialId"] = best["trialId"]
    result["reasonCodes"] = []
    return result


def bqml_valid_evaluate_input(body):
    if not isinstance(body, dict):
        return False
    if body.get("phase") != "evaluate":
        return False
    if not bqml_valid_run_id(body.get("runId")):
        return False
    if not bqml_valid_safe_int(body.get("selectedTrialId")):
        return False

    digest = body.get("datasetDigest")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        return False

    metric_floor = body.get("metricFloor")
    if (
        isinstance(metric_floor, bool)
        or not isinstance(metric_floor, (int, float))
        or not math.isfinite(float(metric_floor))
        or not 0 <= float(metric_floor) <= 1
    ):
        return False

    required_slices = body.get("requiredSlices")
    if not isinstance(required_slices, dict):
        return False

    for name, floor in required_slices.items():
        if not isinstance(name, str) or not name:
            return False
        if (
            isinstance(floor, bool)
            or not isinstance(floor, (int, float))
            or not math.isfinite(float(floor))
            or not 0 <= float(floor) <= 1
        ):
            return False

    rows = body.get("rows")
    if not isinstance(rows, list):
        return False

    if not bqml_valid_safe_int(body.get("bytesProcessed")):
        return False
    if not bqml_valid_safe_int(body.get("maxBytes")):
        return False

    return True


def bqml_valid_test_row(row):
    if not isinstance(row, dict):
        return False
    if set(row.keys()) != {"label", "prediction", "slice"}:
        return False

    if type(row["label"]) is not int or row["label"] not in (0, 1):
        return False
    if type(row["prediction"]) is not int or row["prediction"] not in (0, 1):
        return False
    if not isinstance(row["slice"], str) or not row["slice"]:
        return False

    return True


def bqml_evaluate(body):
    run_id = body.get("runId") if isinstance(body, dict) else ""
    selected = body.get("selectedTrialId") if isinstance(body, dict) else None
    bytes_processed = body.get("bytesProcessed") if isinstance(body, dict) else 0

    digest = body.get("datasetDigest") if isinstance(body, dict) else None
    base = {
        "runId": run_id if isinstance(run_id, str) else "",
        "selectedTrialId": selected if bqml_valid_safe_int(selected) else None,
        "datasetDigest": digest if isinstance(digest, str) else None,
        "testMetric": None,
        "criticalSlicePass": False,
        "decision": "reject",
        "bytesProcessed": bytes_processed if bqml_valid_safe_int(bytes_processed) else 0,
        "reasonCodes": [],
    }

    if not bqml_valid_evaluate_input(body):
        base["reasonCodes"] = ["INVALID_INPUT"]
        return base

    # Preserve the frozen selection digest in the response even when the
    # evaluation fails lineage or gate checks.
    base["datasetDigest"] = body["datasetDigest"]

    with BQML_LOCK:
        stored = BQML_RUNS.get(run_id)

    lineage_ok = bool(
        stored
        and stored["result"].get("selectedTrialId") is not None
        and stored["result"].get("reasonCodes") == []
        and stored["result"].get("selectedTrialId") == selected
        and stored["result"].get("datasetDigest") == body["datasetDigest"]
    )

    reasons = []
    if not lineage_ok:
        reasons.append("INVALID_LINEAGE")

    rows = body["rows"]

    # Empty rows skip aggregate/slice checks, but lineage and byte checks still apply.
    if not rows:
        if body["bytesProcessed"] > body["maxBytes"]:
            reasons.append("BYTE_LIMIT")
        base["criticalSlicePass"] = lineage_ok
        base["reasonCodes"] = bqml_codes(reasons)
        if lineage_ok and body["bytesProcessed"] <= body["maxBytes"]:
            base["decision"] = "admit"
        else:
            base["decision"] = "reject"
        return base

    all_valid = all(bqml_valid_test_row(row) for row in rows)

    if not all_valid:
        reasons.append("INVALID_TEST_ROW")
        if body["bytesProcessed"] > body["maxBytes"]:
            reasons.append("BYTE_LIMIT")
        base["criticalSlicePass"] = False
        base["reasonCodes"] = bqml_codes(reasons)
        base["decision"] = "reject"
        return base

    correct = sum(
        1 for row in rows
        if row["label"] == row["prediction"]
    )
    test_metric = round(correct / len(rows), 12)
    base["testMetric"] = test_metric

    if test_metric < float(body["metricFloor"]):
        reasons.append("AGGREGATE_FLOOR")

    by_slice = {}
    for row in rows:
        by_slice.setdefault(row["slice"], []).append(row)

    slice_pass = True
    required = body["requiredSlices"]

    for name in sorted(required.keys(), key=bqml_utf8):
        if name not in by_slice:
            reasons.append(f"MISSING_SLICE:{name}")
            slice_pass = False
            continue

        slice_rows = by_slice[name]
        slice_correct = sum(
            1 for row in slice_rows
            if row["label"] == row["prediction"]
        )
        accuracy = round(slice_correct / len(slice_rows), 12)

        if accuracy < float(required[name]):
            reasons.append(f"SLICE_FLOOR:{name}")
            slice_pass = False

    if body["bytesProcessed"] > body["maxBytes"]:
        reasons.append("BYTE_LIMIT")

    # This field deliberately does not summarize aggregate or byte gates.
    base["criticalSlicePass"] = (lineage_ok and slice_pass)
    base["reasonCodes"] = bqml_codes(reasons)

    if (
        lineage_ok
        and test_metric >= float(body["metricFloor"])
        and slice_pass
        and body["bytesProcessed"] <= body["maxBytes"]
    ):
        base["decision"] = "admit"
    else:
        base["decision"] = "reject"

    return base


@app.post("/bqml")
async def bqml(request: Request):
    try:
        body = await request.json()
    except Exception:
        return bqml_error(400, "INVALID_INPUT")

    if not isinstance(body, dict):
        return bqml_error(400, "INVALID_INPUT")

    phase = body.get("phase")

    if phase not in ("select", "evaluate"):
        return bqml_error(400, "INVALID_INPUT")

    if phase == "select":
        run_id = body.get("runId")

        if bqml_valid_run_id(run_id):
            fingerprint = bqml_selection_fingerprint(body)

            with BQML_LOCK:
                existing = BQML_RUNS.get(run_id)

            if existing is not None:
                if existing["fingerprint"] == fingerprint:
                    return Response(
                        content=existing["serialized"],
                        media_type="application/json",
                    )

                return bqml_error(409, "RUN_ID_CONFLICT")

        result = bqml_build_selection(body)
        serialized = bqml_json(result)

        # Persist every syntactically valid runId, including failed selections.
        # This makes identical replay deterministic and makes reuse with changed
        # selection input a conflict.
        if bqml_valid_run_id(run_id):
            with BQML_LOCK:
                BQML_RUNS[run_id] = {
                    "fingerprint": bqml_selection_fingerprint(body),
                    "result": result,
                    "serialized": serialized,
                }

        return Response(
            content=serialized,
            media_type="application/json",
        )

    result = bqml_evaluate(body)
    return Response(
        content=bqml_json(result),
        media_type="application/json",
    )


# ---------------------------------------------------------------------------
# Deterministic model-registry promotion gate
# ---------------------------------------------------------------------------

PROMOTE_ALIAS_STATE = {"version": None}


def promote_json(obj):
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def promote_valid_safe_int(value, positive=False):
    if isinstance(value, bool) or not isinstance(value, int):
        return False
    if positive:
        return 1 <= value <= SAFE_INT_MAX
    return 0 <= value <= SAFE_INT_MAX


def promote_valid_canonical_version(value):
    if not isinstance(value, str):
        return None
    if not value or not re.fullmatch(r"[1-9][0-9]*", value):
        return None
    try:
        num = int(value)
    except ValueError:
        return None
    if num <= 0 or num > SAFE_INT_MAX:
        return None
    return str(num)


def promote_rate(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    f = float(value)
    if not math.isfinite(f):
        return None
    if not 0.0 <= f <= 1.0:
        return None
    return f


def promote_parse_policy(policy):
    if not isinstance(policy, dict):
        return None, "INVALID_POLICY"

    required_keys = {
        "datasetDigest",
        "schemaDigest",
        "maxAgeSeconds",
        "accuracyFloor",
        "requiredSlices",
        "maxLatencyMs",
        "maxSizeBytes",
        "minImprovement",
    }
    if not required_keys.issubset(policy.keys()):
        return None, "INVALID_POLICY"

    dataset_digest = policy.get("datasetDigest")
    schema_digest = policy.get("schemaDigest")
    if not isinstance(dataset_digest, str) or not dataset_digest:
        return None, "INVALID_POLICY"
    if not isinstance(schema_digest, str) or not schema_digest:
        return None, "INVALID_POLICY"

    max_age = policy.get("maxAgeSeconds")
    if not promote_valid_safe_int(max_age, positive=False):
        return None, "INVALID_POLICY"

    accuracy_floor = promote_rate(policy.get("accuracyFloor"))
    if accuracy_floor is None:
        return None, "INVALID_POLICY"

    required_slices = policy.get("requiredSlices")
    if not isinstance(required_slices, dict):
        return None, "INVALID_POLICY"

    normalized_slices = {}
    for name, floor in required_slices.items():
        if not isinstance(name, str) or not name:
            return None, "INVALID_POLICY"
        val = promote_rate(floor)
        if val is None:
            return None, "INVALID_POLICY"
        normalized_slices[name] = val

    max_latency = policy.get("maxLatencyMs")
    if isinstance(max_latency, bool) or not isinstance(max_latency, (int, float)):
        return None, "INVALID_POLICY"
    if not math.isfinite(float(max_latency)) or float(max_latency) < 0:
        return None, "INVALID_POLICY"

    max_size = policy.get("maxSizeBytes")
    if not promote_valid_safe_int(max_size, positive=False):
        return None, "INVALID_POLICY"

    min_improvement = promote_rate(policy.get("minImprovement"))
    if min_improvement is None:
        return None, "INVALID_POLICY"

    return {
        "datasetDigest": dataset_digest,
        "schemaDigest": schema_digest,
        "maxAgeSeconds": int(max_age),
        "accuracyFloor": accuracy_floor,
        "requiredSlices": normalized_slices,
        "maxLatencyMs": float(max_latency),
        "maxSizeBytes": int(max_size),
        "minImprovement": min_improvement,
    }, None


def promote_sort_codes(codes):
    return sorted(set(codes), key=bqml_utf8)


def promote_failed_gate_map(version_key, codes):
    return codes


@app.post("/promote")
async def promote(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "INVALID_INPUT"}, status_code=400)

    if not isinstance(body, dict):
        return JSONResponse({"error": "INVALID_INPUT"}, status_code=400)

    as_of_raw = body.get("asOf")
    champion_raw = body.get("championVersion")
    policy_raw = body.get("policy")
    versions_raw = body.get("versions")

    if not isinstance(champion_raw, str):
        return JSONResponse({"error": "INVALID_INPUT"}, status_code=400)
    if not isinstance(versions_raw, list):
        return JSONResponse({"error": "INVALID_INPUT"}, status_code=400)
    if not isinstance(policy_raw, dict):
        return JSONResponse({"error": "INVALID_INPUT"}, status_code=400)
    if parse_event_time(as_of_raw) is None:
        return JSONResponse({"error": "INVALID_INPUT"}, status_code=400)

    compact_result = {
        "action": "block",
        "championVersion": champion_raw,
        "selectedVersion": None,
        "eligibleVersions": [],
        "failedGates": {},
        "aliasMutation": None,
        "evidence": None,
    }

    policy, policy_error = promote_parse_policy(policy_raw)
    if policy_error is not None:
        return JSONResponse({"error": "INVALID_INPUT"}, status_code=400)

    as_of_dt = parse_event_time(as_of_raw)
    assert as_of_dt is not None

    failed_gates = {}
    valid_versions = {}
    seen_versions = set()

    for item in versions_raw:
        if not isinstance(item, dict):
            key = "__invalid__"
            failed_gates.setdefault(key, set()).add("INVALID_VERSION")
            continue

        version_raw = item.get("version")
        version_key = version_raw if isinstance(version_raw, str) else str(version_raw)
        norm_version = promote_valid_canonical_version(version_raw)
        if norm_version is None:
            failed_gates.setdefault(version_key if isinstance(version_raw, str) else "__invalid__", set()).add("INVALID_VERSION")
            continue

        if norm_version in seen_versions:
            failed_gates.setdefault(norm_version, set()).add("DUPLICATE_VERSION")
            continue
        seen_versions.add(norm_version)
        valid_versions[norm_version] = item

    def add_gate(version_key, code):
        failed_gates.setdefault(version_key, set()).add(code)

    for version_str, item in valid_versions.items():
        if not isinstance(item, dict):
            add_gate(version_str, "INVALID_VERSION")
            continue

        evaluation = item.get("evaluation")
        if not isinstance(evaluation, dict):
            add_gate(version_str, "MISSING_EVALUATION")
            continue

        version_artifact = item.get("artifactDigest")
        if not isinstance(version_artifact, str) or not version_artifact:
            add_gate(version_str, "INVALID_VERSION")
            continue

        created_at = evaluation.get("createdAt")
        created_dt = parse_event_time(created_at)
        if created_dt is None:
            add_gate(version_str, "INVALID_TIMESTAMP")
            continue
        if created_dt > as_of_dt:
            add_gate(version_str, "FUTURE_EVALUATION")
        if created_dt < as_of_dt - timedelta(seconds=int(policy["maxAgeSeconds"])):
            add_gate(version_str, "STALE_EVALUATION")

        accuracy = evaluation.get("accuracy")
        acc_val = promote_rate(accuracy)
        if acc_val is None:
            add_gate(version_str, "NON_FINITE")
        elif not 0.0 <= acc_val <= 1.0:
            add_gate(version_str, "METRIC_RANGE")
        if acc_val is not None and acc_val < policy["accuracyFloor"]:
            add_gate(version_str, "ACCURACY_FLOOR")

        latency = evaluation.get("latencyMs")
        lat_val = None
        if isinstance(latency, bool) or not isinstance(latency, (int, float)):
            add_gate(version_str, "NON_FINITE")
        else:
            if not math.isfinite(float(latency)) or float(latency) < 0:
                add_gate(version_str, "NON_FINITE")
            else:
                lat_val = float(latency)
                if lat_val > policy["maxLatencyMs"]:
                    add_gate(version_str, "LATENCY_LIMIT")

        size = evaluation.get("sizeBytes")
        size_val = None
        if isinstance(size, bool) or not isinstance(size, (int, float)):
            add_gate(version_str, "NON_FINITE")
        else:
            if not math.isfinite(float(size)) or float(size) < 0:
                add_gate(version_str, "NON_FINITE")
            else:
                size_val = float(size)
                if size_val > policy["maxSizeBytes"]:
                    add_gate(version_str, "SIZE_LIMIT")

        eval_artifact = evaluation.get("artifactDigest")
        if not isinstance(eval_artifact, str) or not eval_artifact:
            add_gate(version_str, "ARTIFACT_MISMATCH")
        elif eval_artifact != version_artifact:
            add_gate(version_str, "ARTIFACT_MISMATCH")

        eval_dataset = evaluation.get("datasetDigest")
        if eval_dataset != policy["datasetDigest"]:
            add_gate(version_str, "DATASET_MISMATCH")

        eval_schema = evaluation.get("schemaDigest")
        if eval_schema != policy["schemaDigest"]:
            add_gate(version_str, "SCHEMA_MISMATCH")

        slices = evaluation.get("slices")
        if not isinstance(slices, dict):
            for name in sorted(policy["requiredSlices"].keys(), key=bqml_utf8):
                add_gate(version_str, f"MISSING_SLICE:{name}")
        else:
            for name, floor in sorted(policy["requiredSlices"].items(), key=lambda kv: bqml_utf8(kv[0])):
                if name not in slices:
                    add_gate(version_str, f"MISSING_SLICE:{name}")
                    continue
                value = slices[name]
                val = promote_rate(value)
                if val is None:
                    add_gate(version_str, f"SLICE_RANGE:{name}")
                    continue
                if val < floor:
                    add_gate(version_str, f"SLICE_FLOOR:{name}")

    eligible = []
    version_to_eval = {}
    for version_str, item in valid_versions.items():
        gates = promote_sort_codes(failed_gates.get(version_str, set()))
        if not gates:
            evaluation = item.get("evaluation")
            if isinstance(evaluation, dict):
                acc = promote_rate(evaluation.get("accuracy"))
                lat = evaluation.get("latencyMs")
                sz = evaluation.get("sizeBytes")
                if acc is not None and lat is not None and sz is not None:
                    if isinstance(lat, bool) or not isinstance(lat, (int, float)):
                        continue
                    if isinstance(sz, bool) or not isinstance(sz, (int, float)):
                        continue
                    if not math.isfinite(float(lat)) or not math.isfinite(float(sz)):
                        continue
                    version_to_eval[version_str] = evaluation
                    eligible.append({
                        "version": version_str,
                        "accuracy": float(acc),
                        "latency": float(lat),
                        "size": float(sz),
                        "evidence": evaluation,
                    })

    eligible.sort(key=lambda item: (-item["accuracy"], item["latency"], item["size"], int(item["version"])))

    compact_result["eligibleVersions"] = [item["version"] for item in eligible]
    for version_str in sorted(failed_gates.keys(), key=bqml_utf8):
        compact_result["failedGates"][version_str] = promote_sort_codes(failed_gates[version_str])

    champion_version = champion_raw
    champion_data = None
    champion_entry = valid_versions.get(champion_version)
    champion_valid = champion_version in version_to_eval

    if not champion_valid:
        compact_result["action"] = "block"
        compact_result["selectedVersion"] = None
        compact_result["evidence"] = None
        compact_result["aliasMutation"] = None
        return Response(content=promote_json(compact_result), media_type="application/json")

    champion_data = version_to_eval[champion_version]
    if PROMOTE_ALIAS_STATE["version"] is not None and PROMOTE_ALIAS_STATE["version"] in version_to_eval and PROMOTE_ALIAS_STATE["version"] != champion_version:
        selected = PROMOTE_ALIAS_STATE["version"]
        compact_result["action"] = "retain"
        compact_result["selectedVersion"] = selected
        compact_result["evidence"] = version_to_eval[selected]
        compact_result["aliasMutation"] = None
        return Response(content=promote_json(compact_result), media_type="application/json")

    challenger = None
    for item in eligible:
        if item["version"] == champion_version:
            continue
        challenger = item
        break

    if challenger is None:
        compact_result["action"] = "retain"
        compact_result["selectedVersion"] = champion_version
        compact_result["evidence"] = champion_data
        compact_result["aliasMutation"] = None
        return Response(content=promote_json(compact_result), media_type="application/json")

    improvement = round(challenger["accuracy"] - champion_data["accuracy"], 12)
    if improvement >= policy["minImprovement"]:
        compact_result["action"] = "promote"
        compact_result["selectedVersion"] = challenger["version"]
        compact_result["evidence"] = challenger["evidence"]
        compact_result["aliasMutation"] = {"alias": "champion", "version": challenger["version"]}
        PROMOTE_ALIAS_STATE["version"] = challenger["version"]
    else:
        compact_result["action"] = "retain"
        compact_result["selectedVersion"] = champion_version
        compact_result["evidence"] = champion_data
        compact_result["aliasMutation"] = None

    return Response(content=promote_json(compact_result), media_type="application/json")
