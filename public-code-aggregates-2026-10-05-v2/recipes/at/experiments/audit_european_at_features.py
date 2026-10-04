"""Non-price qualification of the fixed AT workbook, not a predictive experiment.

Skip Price/Additional cells BEFORE XML value decoding. Never emit row values,
titles, individual fingerprints, prices, or a fitted-model result. Raw source
bytes are accessible locally; this is access discipline, not third-party blinding.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path

from audit_european_at_schema import (
    AT_BYTES, AT_SHA256, HEADER_FIELDS, probe, recognized_header, workbook_sheets,
)
from audit_turkey_feature_only import (
    CELL, cell_reference, decode_approved_cell, fragments,
    resolve_shared_strings, row_number,
)

BLOCKED_FIELDS = frozenset({"price", "additional"})
NUMERIC_FIELDS = frozenset({"year", "mileage_km", "power_ps", "power_kw",
                            "engine_volume_unit_unverified"})


def rows(archive, member, header, counters):
    """Read only exact mapped non-price columns; other payloads remain bytes."""
    fields = {col: HEADER_FIELDS[label] for col, label in header.items()}
    approved = {col for col, field in fields.items() if field not in BLOCKED_FIELDS}
    last = 1
    with archive.open(member) as stream:
        for block in fragments(stream, b"row"):
            number = row_number(block)
            if number == 1:
                continue
            if number <= last:
                raise ValueError("Data row coordinates must be strictly increasing")
            last = number
            record, seen = {}, set()
            for cell in CELL.findall(block):
                col, row = cell_reference(cell)
                if row != number or col in seen:
                    raise ValueError("Unexpected data-cell coordinates")
                seen.add(col)
                # January has an unlabelled G data column. Its meaning cannot
                # be inferred from values; quarantine it before any decoder.
                if col not in header:
                    counters["unlabelled_data_cells_skipped"] += 1
                    continue
                if col not in approved:
                    counters["blocked_data_cells_skipped"] += 1
                    continue
                record[fields[col]] = decode_approved_cell(cell, col, approved)
                counters["approved_data_cells_decoded"] += 1
            yield number, record


def clean(value: str) -> str:
    return " ".join(value.replace("\xa0", " ").strip().casefold().split())


def number(value: str, field: str) -> Decimal | None:
    """Explicit unit-aware numeric grammar; no sample-value fallback/inference."""
    text = clean(value)
    suffix = {"mileage_km": r"\s*km", "power_ps": r"\s*(?:ps|hp)",
              "power_kw": r"\s*kw"}.get(field)
    if suffix:
        text = re.sub(suffix + r"$", "", text)
    text = text.replace(" ", "").replace("'", "")
    if field in {"mileage_km", "year"}:
        if re.fullmatch(r"\d{1,3}(?:[.,]\d{3})+", text):
            text = text.replace(".", "").replace(",", "")
        elif re.fullmatch(r"\d+[.,]0+", text):
            text = text.replace(",", ".")
        elif not re.fullmatch(r"\d+", text):
            return None
    else:
        if not re.fullmatch(r"\d+(?:[.,]\d+)?", text):
            return None
        text = text.replace(",", ".")
    try:
        result = Decimal(text)
    except InvalidOperation:
        return None
    return result if result.is_finite() and result >= 0 else None


def power_pair(values):
    """Use explicit PS or kW; keep disagreement visible, not silently impute."""
    ps, kw = number(values.get("power_ps", ""), "power_ps"), number(values.get("power_kw", ""), "power_kw")
    both = ps is not None and kw is not None and ps > 0 and kw > 0
    conflict = both and abs(float(kw) - float(ps) * 0.73549875) > max(2.0, float(kw) * .03)
    valid = (ps is not None and ps > 0) or (kw is not None and kw > 0)
    return valid, bool(conflict)


def classification(value, kind):
    text = clean(value)
    if not text:
        return "blank"
    if kind == "currency":
        return "EUR" if text in {"eur", "€", "euro", "euros"} else "unrecognized"
    if kind == "fuel":
        if "hybrid" in text:
            return "hybrid"
        for pattern, label in ((r"diesel", "diesel"), (r"benzin|petrol|gasoline", "petrol"),
                               (r"elektr|electr", "electric"), (r"erdgas|cng", "cng"),
                               (r"autogas|lpg", "lpg")):
            if re.search(pattern, text):
                return label
        return "unrecognized"
    if kind == "transmission":
        if re.search(r"automatik|automatic|dsg|dct|cvt", text):
            return "automatic"
        if re.search(r"schalt|manuell|manual", text):
            return "manual"
        return "unrecognized"
    raise ValueError("Unsupported safe classification")


def fingerprint(values, fields):
    # Do not write the digest or source values to public/aggregate outputs.
    encoded = json.dumps([clean(values.get(f, "")) for f in fields], ensure_ascii=False).encode()
    return hashlib.sha256(encoded).digest()


def eligible(values):
    year, km = number(values.get("year", ""), "year"), number(values.get("mileage_km", ""), "mileage_km")
    valid_power, conflict = power_pair(values)
    return (bool(clean(values.get("name", ""))) and year is not None and
            Decimal(1900) <= year <= Decimal(2023) and km is not None and valid_power and not conflict and
            classification(values.get("fuel", ""), "fuel") in {"diesel", "petrol", "electric", "hybrid", "cng", "lpg"} and
            classification(values.get("transmission", ""), "transmission") in {"manual", "automatic"} and
            classification(values.get("currency", ""), "currency") == "EUR")


def audit_archive(archive):
    sheets = workbook_sheets(archive)
    headers = {month: recognized_header(archive, member, month) for month, member in sheets}
    needed, first_pass = set(), Counter()
    for month, member in sheets:
        for _, record in rows(archive, member, headers[month], first_pass):
            needed.update(v for kind, v in record.values() if kind == "shared")
    strings = resolve_shared_strings(archive, needed)
    counters, summaries = Counter(), []
    title_year_months, action_profile_months = defaultdict(set), defaultdict(set)
    seen_title_year, seen_profile = set(), set()
    eligible_rows = []  # Internal non-price hashes/months only; never exported.
    for month, member in sheets:
        columns = {HEADER_FIELDS[label] for label in headers[month].values()} - BLOCKED_FIELDS
        field_counts = {f: Counter() for f in columns}
        distinct = {f: set() for f in columns}
        count, eligible_count, month_seen_title, month_seen_profile = 0, 0, set(), set()
        flags = Counter()
        for _, raw in rows(archive, member, headers[month], counters):
            values = {f: (strings[raw[f][1]] if raw[f][0] == "shared" else raw[f][1])
                      if f in raw else "" for f in columns}
            count += 1
            for f, v in values.items():
                v = clean(v)
                field_counts[f]["blank" if not v else "nonblank"] += 1
                if v:
                    distinct[f].add(v)
                if f in NUMERIC_FIELDS and v:
                    field_counts[f]["numeric_parse_success" if number(v, f) is not None else "numeric_parse_failure"] += 1
                if f in {"currency", "fuel", "transmission"}:
                    field_counts[f]["class_" + classification(v, f)] += 1
            year, km = number(values.get("year", ""), "year"), number(values.get("mileage_km", ""), "mileage_km")
            valid_power, conflict = power_pair(values)
            flags["power_pair_unit_conflict_rows"] += int(conflict)
            flags["both_ps_and_kw_nonblank_rows"] += int(bool(clean(values.get("power_ps", ""))) and bool(clean(values.get("power_kw", ""))))
            flags["name_nonblank_rows"] += int(bool(clean(values.get("name", ""))))
            is_eligible = eligible(values)
            eligible_count += int(is_eligible)
            if not clean(values.get("name", "")) or year is None:
                continue
            ty = fingerprint(values, ("name", "year"))
            profile = fingerprint(values, ("name", "year", "mileage_km", "power_ps", "power_kw", "fuel", "transmission"))
            title_year_months[ty].add(month)
            action_profile_months[profile].add(month)
            month_seen_title.add(ty)
            month_seen_profile.add(profile)
            if is_eligible:
                eligible_rows.append((month, ty, profile))
        summaries.append({
            "month": month, "data_rows": count, "fields": {
                f: {**dict(field_counts[f]), "distinct_nonblank": len(distinct[f])}
                for f in sorted(columns)},
            "flags": dict(flags), "joint_four_action_nonprice_eligible_rows": eligible_count,
            "distinct_title_year_fingerprints": len(month_seen_title),
            "title_year_fingerprints_seen_in_earlier_months": len(month_seen_title & seen_title_year),
            "distinct_action_profile_fingerprints": len(month_seen_profile),
            "action_profile_fingerprints_seen_in_earlier_months": len(month_seen_profile & seen_profile),
        })
        seen_title_year.update(month_seen_title)
        seen_profile.update(month_seen_profile)
    # Feasibility only, not a selected evaluation split. Groups spanning a prior
    # role cannot enter later-role evaluation. Each key counted once per role.
    roles = {"development": {f"{m:02}" for m in range(2, 8)},
             "validation": {"08"}, "calibration": {"09", "10"}, "evaluation": {"11", "12"}}
    feasibility = {}
    for key_index, name in ((1, "title_year"), (2, "action_profile")):
        seen, per_role = set(), {}
        for role, months in roles.items():
            candidates = {r[key_index] for r in eligible_rows if r[0] in months}
            accepted = candidates - seen
            per_role[role] = {"candidate_unique_nonprice_groups": len(candidates),
                              "groups_seen_in_prior_roles": len(candidates & seen),
                              "earliest_role_disjoint_groups": len(accepted)}
            seen.update(candidates)
        feasibility[name] = per_role
    return {
        "stage": "NONPRICE_FEATURE_QUALIFICATION_NO_MODEL_OR_TARGET_READ",
        "months": summaries, "total_data_rows": sum(m["data_rows"] for m in summaries),
        "cell_access_second_pass": dict(counters), "price_cells_decoded": 0,
        "additional_cells_decoded": 0, "individual_values_or_fingerprints_exported": False,
        "unlabelled_data_cells_decoded": 0,
        "cross_month_title_year_groups": sum(len(ms) > 1 for ms in title_year_months.values()),
        "cross_month_action_profile_groups": sum(len(ms) > 1 for ms in action_profile_months.values()),
        "nonprice_four_action_eligibility_rule": "Name + year 1900..2023 + numeric nonnegative km + positive PS/kW without unit conflict + recognized fuel/transmission + explicit EUR",
        "feasibility_roles_not_final_split": {k: sorted(v) for k, v in roles.items()},
        "earliest_role_unique_group_feasibility": feasibility,
        "fingerprints_are_not_verified_vehicle_ids": True,
        "price_validity_not_checked": True, "months_not_silently_discarded": True,
        "independent_label_custody_or_blinding_claimed": False,
        "model_fit_or_outcome_evaluation_started": False, "experiment_GO": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    probe(args.source)  # Exact hash and all twelve headers before data decoding.
    with zipfile.ZipFile(args.source) as archive:
        result = audit_archive(archive)
    result.update(source_bytes=AT_BYTES, source_sha256=AT_SHA256,
                  raw_price_bytes_machine_received=True,
                  source_workbook_modified_or_exported=False)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"stage": result["stage"], "total_data_rows": result["total_data_rows"],
                      "price_cells_decoded": 0, "experiment_GO": False,
                      "receipt": str(args.receipt)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
