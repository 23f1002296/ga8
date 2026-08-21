from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from typing import Any
from datetime import datetime, timezone, timedelta
from decimal import Decimal
import hashlib
import json
import math
import re
import threading


app = FastAPI()

# Stateful storage.
# Render should run a single instance for the grader.
RUNS: dict[str, dict[str, Any]] = {}
LOCK = threading.Lock()

SAFE_INT_MAX = 9007199254740991

TIMESTAMP_RE = re.compile(
    r"^"
    r"(\d{4})-(\d{2})-(\d{2})"
    r"T"
    r"(\d{2}):(\d{2}):(\d{2})"
    r"(?:\.(\d{1,3}))?"
    r"(Z|[+-]\d{2}:\d{2})"
    r"$"
)

HEX64_RE = re.compile(r"^[0-9a-f]{64}$")


# ----------------------------------------------------------------------
# Deterministic helpers
# ----------------------------------------------------------------------

def is_safe_int(v: Any) -> bool:
    return (
        isinstance(v, int)
        and not isinstance(v, bool)
        and 0 <= v <= SAFE_INT_MAX
    )


def is_positive_safe_int(v: Any) -> bool:
    return (
        isinstance(v, int)
        and not isinstance(v, bool)
        and 1 <= v <= SAFE_INT_MAX
    )


def finite_number(v: Any) -> bool:
    if isinstance(v, bool):
        return False
    if not isinstance(v, (int, float)):
        return False
    return math.isfinite(float(v))


def parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None

    m = TIMESTAMP_RE.fullmatch(value)
    if not m:
        return None

    year = int(m.group(1))
    month = int(m.group(2))
    day = int(m.group(3))
    hour = int(m.group(4))
    minute = int(m.group(5))
    second = int(m.group(6))
    fraction = m.group(7)
    offset = m.group(8)

    # Explicitly validate offset magnitude.
    if offset == "Z":
        tz = timezone.utc
    else:
        sign = 1 if offset[0] == "+" else -1
        oh = int(offset[1:3])
        om = int(offset[4:6])

        if oh > 14:
            return None
        if oh == 14 and om != 0:
            return None

        total_minutes = sign * (oh * 60 + om)
        tz = timezone(
    timedelta(minutes=total_minutes)
)

    try:
        micro = 0
        if fraction:
            micro = int(fraction.ljust(3, "0")) * 1000

        dt = datetime(
            year,
            month,
            day,
            hour,
            minute,
            second,
            microsecond=micro,
            tzinfo=tz,
        )
    except Exception:
        return None

    return dt.astimezone(timezone.utc)


def utc_string(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + \
        f"{dt.microsecond // 1000:03d}Z"


def utf8_key(value: str) -> bytes:
    return value.encode("utf-8")


def compact_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def compact_bytes(value: Any) -> bytes:
    return compact_json(value).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(compact_bytes(value)).hexdigest()


def sorted_utf8(values):
    return sorted(values, key=utf8_key)


def unique_reason_codes(codes):
    return sorted(
        set(codes),
        key=utf8_key,
    )


def round12(value: float) -> float:
    # Decimal(str(...)) avoids binary representation surprises.
    return float(Decimal(str(value)).quantize(
        Decimal("0.000000000001")
    ))


def request_fingerprint(obj: dict[str, Any]) -> str:
    """
    Fingerprint the selection input semantically, not according to
    incoming JSON key order.
    """
    data = compact_json(
        obj
    )
    # For conflict detection, object-key order should not matter.
    data = json.dumps(
        obj,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------
# Input validation
# ----------------------------------------------------------------------

def validate_run_id(v: Any) -> bool:
    return (
        isinstance(v, str)
        and not isinstance(v, bool)
        and 0 < len(v) <= 128
    )


def validate_selection_row(row: Any) -> bool:
    if not isinstance(row, dict):
        return False

    required = {
        "id",
        "entity",
        "eventTime",
        "predictionTime",
        "version",
        "split",
        "features",
    }

    if set(row.keys()) != required:
        return False

    if not isinstance(row["id"], str):
        return False

    if not isinstance(row["entity"], str):
        return False

    if parse_timestamp(row["eventTime"]) is None:
        return False

    if parse_timestamp(row["predictionTime"]) is None:
        return False

    if not is_safe_int(row["version"]):
        return False

    if row["split"] not in ("TRAIN", "EVAL"):
        return False

    if not isinstance(row["features"], dict):
        return False

    for name, feature in row["features"].items():
        if not isinstance(name, str):
            return False

        if not isinstance(feature, dict):
            return False

        # A feature must contain exactly value + availableAt.
        if set(feature.keys()) != {"value", "availableAt"}:
            return False

        if parse_timestamp(feature["availableAt"]) is None:
            return False

    return True


def validate_selection(req: dict[str, Any]):
    if not isinstance(req, dict):
        return False

    if req.get("phase") != "select":
        return False

    if not validate_run_id(req.get("runId")):
        return False

    forbidden = req.get("forbiddenFeatures")
    if not isinstance(forbidden, list):
        return False

    if any(not isinstance(x, str) for x in forbidden):
        return False

    if not is_positive_safe_int(req.get("numTrialsLimit")):
        return False

    rows = req.get("rows")
    if not isinstance(rows, list) or len(rows) == 0:
        return False

    trials = req.get("trials")
    if not isinstance(trials, list):
        return False

    # Row IDs must be unique.
    ids = []
    for row in rows:
        if not validate_selection_row(row):
            return False
        ids.append(row["id"])

    if len(ids) != len(set(ids)):
        return False

    trial_ids = []

    for trial in trials:
        if not isinstance(trial, dict):
            return False

        if set(trial.keys()) != {
            "trialId",
            "status",
            "evalMetric",
        }:
            return False

        if not is_safe_int(trial["trialId"]):
            return False

        if trial["status"] not in ("SUCCEEDED", "FAILED"):
            return False

        if not finite_number(trial["evalMetric"]):
            return False

        trial_ids.append(trial["trialId"])

    if len(trial_ids) != len(set(trial_ids)):
        return False

    if len(trials) > req["numTrialsLimit"]:
        return "TRIAL_LIMIT_EXCEEDED"

    return True


# ----------------------------------------------------------------------
# Row deduplication / point-in-time processing
# ----------------------------------------------------------------------

def deduplicate_rows(rows):
    """
    Dedup key:
        [entity, UTC(eventTime)]

    Winner:
        highest version
        then UTF-8-byte-smallest ID
    """

    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}

    for row in rows:
        event_dt = parse_timestamp(row["eventTime"])
        prediction_dt = parse_timestamp(row["predictionTime"])

        normalized = dict(row)
        normalized["_event_dt"] = event_dt
        normalized["_prediction_dt"] = prediction_dt
        normalized["_event_utc"] = utc_string(event_dt)
        normalized["_prediction_utc"] = utc_string(prediction_dt)

        key = (
            row["entity"],
            normalized["_event_utc"],
        )

        groups.setdefault(key, []).append(normalized)

    retained = []
    losers = []

    for group in groups.values():
        ordered = sorted(
            group,
            key=lambda r: (
                -r["version"],
                utf8_key(r["id"]),
            ),
        )

        winner = ordered[0]
        retained.append(winner)

        for loser in ordered[1:]:
            losers.append(loser)

    retained.sort(key=lambda r: utf8_key(r["id"]))
    losers.sort(key=lambda r: utf8_key(r["id"]))

    return retained, losers


def shared_eligible_features(rows, forbidden):
    if not rows:
        return []

    forbidden_set = set(forbidden)

    # Start with the names present in every retained row.
    common = set(rows[0]["features"].keys())

    for row in rows[1:]:
        common.intersection_update(row["features"].keys())

    eligible = []

    for name in common:
        if name in forbidden_set:
            continue

        available_everywhere = True

        for row in rows:
            available_at = parse_timestamp(
                row["features"][name]["availableAt"]
            )

            prediction_time = row["_prediction_dt"]

            if available_at > prediction_time:
                available_everywhere = False
                break

        if available_everywhere:
            eligible.append(name)

    return sorted_utf8(eligible)


# ----------------------------------------------------------------------
# Selection
# ----------------------------------------------------------------------

def do_selection(req):
    invalid = validate_selection(req)

    if invalid is False:
        return {
            "runId": req.get("runId") if isinstance(
                req.get("runId"), str
            ) else "",
            "selectedTrialId": None,
            "trainRowIds": [],
            "evalRowIds": [],
            "featureNames": [],
            "datasetDigest": None,
            "reasonCodes": ["INVALID_INPUT"],
        }

    if invalid == "TRIAL_LIMIT_EXCEEDED":
        return {
            "runId": req["runId"],
            "selectedTrialId": None,
            "trainRowIds": [],
            "evalRowIds": [],
            "featureNames": [],
            "datasetDigest": None,
            "reasonCodes": ["TRIAL_LIMIT_EXCEEDED"],
        }

    retained, losers = deduplicate_rows(req["rows"])

    # Only retained rows participate in the dataset.
    train_rows = [
        r for r in retained
        if r["split"] == "TRAIN"
    ]

    eval_rows = [
        r for r in retained
        if r["split"] == "EVAL"
    ]

    train_ids = sorted_utf8(
        [r["id"] for r in train_rows]
    )

    eval_ids = sorted_utf8(
        [r["id"] for r in eval_rows]
    )

    feature_names = shared_eligible_features(
        retained,
        req["forbiddenFeatures"],
    )

    # Successful trials only, finite already guaranteed by validation.
    successful = [
        t for t in req["trials"]
        if t["status"] == "SUCCEEDED"
        and finite_number(t["evalMetric"])
    ]

    if not successful:
        return {
            "runId": req["runId"],
            "selectedTrialId": None,
            "trainRowIds": train_ids,
            "evalRowIds": eval_ids,
            "featureNames": feature_names,
            "datasetDigest": sha256_json({
                "trainRowIds": train_ids,
                "evalRowIds": eval_ids,
                "featureNames": feature_names,
            }),
            "reasonCodes": ["NO_SUCCESSFUL_TRIAL"],
        }

    best = max(
        successful,
        key=lambda t: (
            float(t["evalMetric"]),
            -t["trialId"],
        ),
    )

    digest_payload = {
        "trainRowIds": train_ids,
        "evalRowIds": eval_ids,
        "featureNames": feature_names,
    }

    digest = sha256_json(digest_payload)

    return {
        "runId": req["runId"],
        "selectedTrialId": best["trialId"],
        "trainRowIds": train_ids,
        "evalRowIds": eval_ids,
        "featureNames": feature_names,
        "datasetDigest": digest,
        "reasonCodes": [],
    }


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------

def validate_evaluation_request(req):
    if not isinstance(req, dict):
        return False

    if req.get("phase") != "evaluate":
        return False

    if not validate_run_id(req.get("runId")):
        return False

    if not is_safe_int(req.get("selectedTrialId")):
        return False

    digest = req.get("datasetDigest")
    if (
        not isinstance(digest, str)
        or HEX64_RE.fullmatch(digest) is None
    ):
        return False

    if not finite_number(req.get("metricFloor")):
        return False

    metric_floor = float(req["metricFloor"])
    if not 0 <= metric_floor <= 1:
        return False

    required = req.get("requiredSlices")
    if not isinstance(required, dict):
        return False

    for name, floor in required.items():
        if not isinstance(name, str):
            return False
        if not finite_number(floor):
            return False
        if not 0 <= float(floor) <= 1:
            return False

    rows = req.get("rows")
    if not isinstance(rows, list):
        return False

    if not is_safe_int(req.get("bytesProcessed")):
        return False

    if not is_safe_int(req.get("maxBytes")):
        return False

    return True


def validate_test_row(row):
    if not isinstance(row, dict):
        return False

    if set(row.keys()) != {
        "label",
        "prediction",
        "slice",
    }:
        return False

    if row["label"] not in (0, 1):
        return False

    if row["prediction"] not in (0, 1):
        return False

    if not isinstance(row["slice"], str):
        return False

    if len(row["slice"]) == 0:
        return False

    return True


def do_evaluation(req):
    base = {
        "runId": req.get("runId", ""),
        "selectedTrialId": (
            req.get("selectedTrialId")
            if is_safe_int(req.get("selectedTrialId"))
            else None
        ),
        "testMetric": None,
        "criticalSlicePass": False,
        "decision": "reject",
        "bytesProcessed": (
            req.get("bytesProcessed")
            if is_safe_int(req.get("bytesProcessed"))
            else 0
        ),
        "reasonCodes": [],
    }

    if not validate_evaluation_request(req):
        base["reasonCodes"] = ["INVALID_INPUT"]
        return base

    run_id = req["runId"]

    with LOCK:
        stored = RUNS.get(run_id)

    # The lineage must point to a successful frozen selection.
    if stored is None:
        base["reasonCodes"] = ["INVALID_LINEAGE"]
        return base

    selection = stored["response"]

    if (
        selection["selectedTrialId"] is None
        or selection["datasetDigest"] is None
        or selection["selectedTrialId"] != req["selectedTrialId"]
        or selection["datasetDigest"] != req["datasetDigest"]
        or selection["reasonCodes"]
    ):
        base["reasonCodes"] = ["INVALID_LINEAGE"]
        return base

    rows = req["rows"]

    # Empty test set is explicitly invalid for metric computation,
    # but lineage and byte checks still apply.
    if len(rows) == 0:
        base["testMetric"] = None
        base["criticalSlicePass"] = False

        reasons = []

        if req["bytesProcessed"] > req["maxBytes"]:
            reasons.append("BYTE_LIMIT")

        base["reasonCodes"] = unique_reason_codes(reasons)

        # No test rows means reject.
        return base

    invalid_test_row = False

    for row in rows:
        if not validate_test_row(row):
            invalid_test_row = True
            break

    if invalid_test_row:
        base["testMetric"] = None
        base["criticalSlicePass"] = False

        reasons = ["INVALID_TEST_ROW"]

        if req["bytesProcessed"] > req["maxBytes"]:
            reasons.append("BYTE_LIMIT")

        base["reasonCodes"] = unique_reason_codes(reasons)
        return base

    # Aggregate accuracy.
    correct = sum(
        1
        for r in rows
        if r["label"] == r["prediction"]
    )

    aggregate = round12(correct / len(rows))
    base["testMetric"] = aggregate

    # Slice statistics.
    slice_counts: dict[str, int] = {}
    slice_correct: dict[str, int] = {}

    for row in rows:
        name = row["slice"]

        slice_counts[name] = slice_counts.get(name, 0) + 1

        if row["label"] == row["prediction"]:
            slice_correct[name] = slice_correct.get(name, 0) + 1

    required = req["requiredSlices"]

    reasons = []

    # Required slices must exist.
    missing = []

    for name in required:
        if name not in slice_counts:
            missing.append(name)

    for name in sorted_utf8(missing):
        reasons.append(f"MISSING_SLICE:{name}")

    # Aggregate floor.
    if aggregate < float(req["metricFloor"]):
        reasons.append("AGGREGATE_FLOOR")

    # Slice floors.
    slice_pass = True

    for name in sorted_utf8(required.keys()):
        if name not in slice_counts:
            slice_pass = False
            continue

        accuracy = round12(
            slice_correct.get(name, 0) /
            slice_counts[name]
        )

        if accuracy < float(required[name]):
            slice_pass = False
            reasons.append(f"SLICE_FLOOR:{name}")

    # Byte gate.
    if req["bytesProcessed"] > req["maxBytes"]:
        reasons.append("BYTE_LIMIT")

    # criticalSlicePass specifically describes required-slice validity.
    # It does NOT summarize aggregate or byte gates.
    if missing:
        slice_pass = False

    base["criticalSlicePass"] = slice_pass
    base["reasonCodes"] = unique_reason_codes(reasons)

    # Admit only when ALL gates pass.
    if (
        aggregate >= float(req["metricFloor"])
        and slice_pass
        and req["bytesProcessed"] <= req["maxBytes"]
        and not missing
    ):
        base["decision"] = "admit"
    else:
        base["decision"] = "reject"

    return base


# ----------------------------------------------------------------------
# Endpoint
# ----------------------------------------------------------------------

@app.post("/bqml")
async def bqml(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            status_code=400,
            content={"error": "INVALID_INPUT"},
        )

    if not isinstance(body, dict):
        return JSONResponse(
            status_code=400,
            content={"error": "INVALID_INPUT"},
        )

    phase = body.get("phase")

    if phase not in ("select", "evaluate"):
        return JSONResponse(
            status_code=400,
            content={"error": "INVALID_INPUT"},
        )

    # --------------------------------------------------------------
    # SELECT
    # --------------------------------------------------------------
    if phase == "select":
        run_id = body.get("runId")

        # Existing run: identical replay or conflict.
        if isinstance(run_id, str) and validate_run_id(run_id):
            fingerprint = request_fingerprint(body)

            with LOCK:
                existing = RUNS.get(run_id)

            if existing is not None:
                if existing["fingerprint"] == fingerprint:
                    # Exact stored response, unchanged.
                    return JSONResponse(
                        status_code=200,
                        content=existing["response"],
                    )

                return JSONResponse(
                    status_code=409,
                    content={"error": "RUN_ID_CONFLICT"},
                )

        response = do_selection(body)

        # Persist complete response under runId.
        if validate_run_id(body.get("runId")):
            with LOCK:
                RUNS[body["runId"]] = {
                    "fingerprint": request_fingerprint(body),
                    "response": response,
                }

        return JSONResponse(
            status_code=200,
            content=response,
        )

    # --------------------------------------------------------------
    # EVALUATE
    # --------------------------------------------------------------
    response = do_evaluation(body)

    return JSONResponse(
        status_code=200,
        content=response,
    )


@app.get("/")
def health():
    return {"status": "ok"}