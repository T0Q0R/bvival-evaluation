"""Feature-only cohort freeze; never reads or exports price cell values.

Separate private initial and acquisition tables. Source locators/fingerprints
are not model features. Grouping mitigates specified profile overlap, not VIN
or seller leakage. This does not release labels or compute any performance.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import unicodedata
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path

import audit_turkey_feature_only as reader
from archive_turkey_candidate import EXPECTED_SHA256, verify_archive


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def text(value: str, config: dict) -> str:
    normalized = " ".join(unicodedata.normalize("NFKC", value).split()).casefold()
    return "" if normalized in config["missing_tokens_casefold"] else normalized


def number(value: str):
    try:
        parsed = Decimal(value)
    except (InvalidOperation, ValueError):
        return None
    return parsed if parsed.is_finite() else None


def canonical_numeric(value: str) -> str:
    parsed = number(value)
    return format(parsed.normalize(), "f") if parsed is not None else value


def validate_config(config):
    fields = set(config["source_feature_fields"])
    roles = [set(config[k]) for k in ("initial_visible_fields", "action_fields", "excluded_predictor_fields")]
    if (any(reader.BLOCKED_FIELD_NAME.search(f) for f in fields)
            or fields != set(config["expected_headers"]) - {"fiyat"}
            or any(roles[i] & roles[j] for i in range(3) for j in range(i + 1, 3))
            or set.union(*roles) != fields or not {"id", "model", "motor_hacmi", "motor_gucu"}.issubset(roles[2])
            or set(config["action_fields"]) != {"kilometre", "vites_tipi", "yakit_tipi"}
            or set(config["profile_fields"]) != fields - {"id"}
            or not set(config["coarse_group_fields"]).issubset(fields - {"id"})
            or config["split_bucket_ends"] != [["train", 55], ["validation", 70], ["calibration", 85], ["test", 100]]
            or config["oof_folds"] != 5):
        raise ValueError("Invalid price-free feature roles or fixed grouping/split contract")


def read_records(archive, header, config):
    validate_config(config)
    if set(header.values()) != set(config["expected_headers"]):
        raise ValueError("Source schema differs from frozen header contract")
    fields, needed, counters = config["source_feature_fields"], set(), Counter()
    path = reader.sheet_path(archive)
    for record in reader.selected_rows(archive, path, header, fields, counters):
        needed.update(value for kind, value in record.values() if kind == "shared")
    strings = reader.resolve_shared_strings(archive, needed)
    counters = Counter()
    records = []
    for ordinal, record in enumerate(reader.selected_rows(archive, path, header, fields, counters), start=1):
        values = {f: (strings[record[f][1]] if record[f][0] == "shared" else record[f][1]).strip() if f in record else "" for f in fields}
        if values["id"] != str(ordinal):
            raise ValueError("Pinned source ordinal/id relation changed; do not use ID as vehicle identity")
        records.append((ordinal, values))
    return records, dict(counters)


def bucket(group_hash, config):
    slot = int(digest([config["split_namespace"], group_hash])[:8], 16) % 100
    return next(name for name, end in config["split_bucket_ends"] if slot < end)


def fold(group_hash, config):
    return int(digest([config["oof_namespace"], group_hash])[:8], 16) % config["oof_folds"]


def build_records(records, config):
    validate_config(config)
    expected = set(config["source_feature_fields"])
    eligible, excluded, flags, aliases = [], [], Counter(), Counter()
    seen_ordinals, missing_parts = set(), Counter()
    for ordinal, raw in records:
        if set(raw) != expected or not isinstance(ordinal, int) or ordinal < 1 or ordinal in seen_ordinals:
            raise ValueError("Row payload must match nonprice contract with unique positive ordinal")
        seen_ordinals.add(ordinal)
        values = {f: text(v, config) for f, v in raw.items()}
        for f in config["numeric_source_fields"]:
            values[f] = canonical_numeric(values[f])
        year, mileage = number(values["yil"]), number(values["kilometre"])
        bad = []
        if any(not values[f] for f in config["required_identity_fields"]):
            bad.append("missing_core_identity")
        if year is None or year != year.to_integral_value() or not config["year_range_inclusive"][0] <= year <= config["year_range_inclusive"][1]:
            bad.append("invalid_year")
        if mileage is None or not config["mileage_range_inclusive"][0] <= mileage <= config["mileage_range_inclusive"][1]:
            bad.append("invalid_mileage")
        if values["vites_tipi"] not in {text(v, config) for v in config["transmission_categories"]}:
            bad.append("missing_or_unsupported_transmission")
        if values["yakit_tipi"] not in {text(v, config) for v in config["fuel_categories"]}:
            bad.append("missing_or_unsupported_fuel")
        key = digest([EXPECTED_SHA256, "source_data_ordinal", ordinal])
        flags.update(bad)
        if bad:
            excluded.append({"record_key": key, "reason": bad[0]})
            continue
        # Fingerprints describe source attributes before optional-part masking.
        profile = digest({f: values[f] for f in config["profile_fields"]})
        group = digest({f: values[f] for f in config["coarse_group_fields"]})
        for f in ("boyali_sayisi", "degisen_sayisi"):
            parsed = number(values[f])
            if parsed is None or parsed != parsed.to_integral_value() or not config["part_count_range_inclusive"][0] <= parsed <= config["part_count_range_inclusive"][1]:
                values[f] = ""
                missing_parts[f] += 1
        aliases["series_fuel_or_engine_family_flag"] += int(bool(re.search(r"\b(?:dizel|diesel|benzin|lpg|hybrid|hibrit|electric|elektrik|tsi|tdi|hdi|dci|crdi|cdti|tdci|multijet)\b", values["seri"])))
        aliases["series_transmission_flag"] += int(bool(re.search(r"\b(?:dsg|dct|cvt|otomatik|automatic|manuel|manual|tiptronic|edc)\b", values["seri"])))
        eligible.append({"record_key": key, "group_hash": group, "profile_hash": profile,
                         "source_data_ordinal": ordinal, "values": values})
    by_profile = defaultdict(list)
    for row in eligible:
        by_profile[row["profile_hash"]].append(row)
    kept = []
    for rows in by_profile.values():
        ordered = sorted(rows, key=lambda r: r["record_key"])
        kept.append(ordered[0])
        excluded.extend({"record_key": r["record_key"], "reason": "duplicate_nonprice_profile"} for r in ordered[1:])
    for row in kept:
        row["split"] = bucket(row["group_hash"], config)
        row["oof_fold"] = fold(row["group_hash"], config) if row["split"] == "train" else ""
    kept.sort(key=lambda r: r["record_key"])
    excluded.sort(key=lambda r: r["record_key"])
    summary = {"raw_feature_rows": len(records), "eligible_before_profile_dedup": len(eligible),
               "retained_feature_rows": len(kept), "removed_duplicate_profile_rows": len(eligible) - len(kept),
               "pre_joint_filter_flags": dict(flags), "exclusive_exclusion_counts": dict(Counter(r["reason"] for r in excluded)),
               "optional_part_missing_or_invalid_before_dedup": dict(missing_parts),
               "series_alias_flags_before_dedup": dict(aliases), "split_listing_counts": dict(Counter(r["split"] for r in kept)),
               "split_group_counts": {s: len({r["group_hash"] for r in kept if r["split"] == s}) for s in ("train", "validation", "calibration", "test")},
               "train_oof_listing_counts": dict(Counter(r["oof_fold"] for r in kept if r["split"] == "train")),
               "train_oof_group_counts": {str(f): len({r["group_hash"] for r in kept if r["split"] == "train" and r["oof_fold"] == f}) for f in range(5)},
               "nonprice_groups_are_not_verified_vehicle_ids": True, "prices_unparsed": True,
               "final_scored_sample_count_unknown": True, "confirmatory_test_allowed_now": False}
    if len(kept) + len(excluded) != len(records):
        raise AssertionError("Cohort accounting does not reconcile")
    assert_group_boundaries(kept)
    return kept, excluded, summary


def assert_group_boundaries(rows):
    assignments = defaultdict(set)
    folds = defaultdict(set)
    for row in rows:
        assignments[row["group_hash"]].add(row["split"])
        if row["split"] == "train":
            folds[row["group_hash"]].add(row["oof_fold"])
    if any(len(v) != 1 for v in assignments.values()) or any(len(v) != 1 for v in folds.values()):
        raise AssertionError("A profile group crosses split or OOF boundary")


def write_csv(path, columns, rows):
    with path.open("x", encoding="utf-8", newline="") as stream:
        path.chmod(0o600)
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    verified = verify_archive(args.source)
    validate_config(config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        raise ValueError("Output directory must be empty; never overwrite a frozen cohort")
    args.output_dir.chmod(0o700)
    with zipfile.ZipFile(args.source) as archive:
        header = reader.read_schema(archive, reader.sheet_path(archive))
        records, counters = read_records(archive, header, config)
    kept, excluded, summary = build_records(records, config)
    base = ["record_key", "group_hash", "profile_hash", "split", "oof_fold"]
    write_csv(args.output_dir / "split_manifest.csv", base, [{k: r[k] for k in base} for r in kept])
    visible, actions = config["initial_visible_fields"], config["action_fields"]
    write_csv(args.output_dir / "initial_features.csv", ["record_key", *visible],
              [{"record_key": r["record_key"], **{k: r["values"][k] for k in visible}} for r in kept])
    write_csv(args.output_dir / "acquisition_fields.csv", ["record_key", *actions],
              [{"record_key": r["record_key"], **{k: r["values"][k] for k in actions}} for r in kept])
    write_csv(args.output_dir / "source_locators.csv", ["record_key", "source_data_ordinal"],
              [{k: r[k] for k in ("record_key", "source_data_ordinal")} for r in kept])
    write_csv(args.output_dir / "excluded_records.csv", ["record_key", "reason"], excluded)
    snapshots = {"cohort_builder_snapshot.py": Path(__file__), "reader_snapshot.py": Path(reader.__file__),
                 "archive_helper_snapshot.py": Path(reader.__file__).with_name("archive_turkey_candidate.py"),
                 "config_snapshot.json": args.config, "prelabel_protocol_snapshot.md": args.protocol}
    for name, path in snapshots.items():
        (args.output_dir / name).write_bytes(path.read_bytes())
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.output_dir.iterdir())}
    summary.update({"stage": "PRICE_FREE_COHORT_FREEZE_ONLY", "checked_at_utc": datetime.now(timezone.utc).isoformat(),
                    "source_verified": verified, "cell_access": counters, "price_cell_values_decoded": 0,
                    "models_trained": False, "test_scores_computed": False, "labels_exported": False,
                    "initial_visible_fields": visible, "action_fields": actions,
                    "excluded_predictor_fields": config["excluded_predictor_fields"], "output_sha256": hashes})
    (args.output_dir / "cohort_freeze_audit.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("stage", "raw_feature_rows", "retained_feature_rows", "split_listing_counts", "split_group_counts", "price_cell_values_decoded")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
