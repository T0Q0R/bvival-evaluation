"""Build a private price-free cohort using the audited AT fields only."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from collections import Counter
from decimal import Decimal
from pathlib import Path

from audit_european_at_schema import AT_SHA256, HEADER_FIELDS, probe, recognized_header, workbook_sheets
from audit_european_at_features import rows, clean, eligible, number, fingerprint, classification
from audit_turkey_feature_only import resolve_shared_strings

ROLE_MONTHS = {"development": tuple(f"{m:02}" for m in range(2, 8)),
               "validation": ("08",), "calibration": ("09", "10"), "evaluation": ("11", "12")}
ACTIONS = ("mileage_km", "power_kw", "fuel", "transmission")
# Conservative string-reduction rules, not a learned taxonomy. No free title,
# engine suffix, fuel/gearbox name or trim token survives into policy contexts.
BRANDS = {
    "mercedes-benz": "mercedes", "mercedes benz": "mercedes", "mercedes": "mercedes",
    "volkswagen": "vw", "vw": "vw", "ford": "ford", "renault": "renault",
    "fiat": "fiat", "peugeot": "peugeot", "citroën": "citroen", "citroen": "citroen",
    "opel": "opel", "iveco": "iveco", "man": "man", "mitsubishi": "mitsubishi",
    "fuso": "fuso", "toyota": "toyota", "nissan": "nissan", "isuzu": "isuzu",
    "daf": "daf", "scania": "scania", "volvo": "volvo", "hyundai": "hyundai",
    "kia": "kia", "dacia": "dacia", "skoda": "skoda", "škoda": "skoda",
    "audi": "audi", "bmw": "bmw", "suzuki": "suzuki", "ssangyong": "ssangyong",
}
FAMILIES = {
    "mercedes": ("sprinter", "vito", "viano", "citan", "actros", "atego", "axor", "arocs", "unimog"),
    "vw": ("transporter", "crafter", "caddy", "amarok", "lt", "t4", "t5", "t6", "t6.1", "t7"),
    "ford": ("transit", "ranger", "connect", "courier"),
    "renault": ("master", "trafic", "kangoo", "maxity"),
    "fiat": ("ducato", "doblo", "doblò", "scudo", "talento", "fiorino", "fullback"),
    "peugeot": ("boxer", "expert", "partner", "bipper"),
    "citroen": ("jumper", "jumpy", "berlingo", "nemo"),
    "opel": ("movano", "vivaro", "combo"),
    "iveco": ("daily", "stralis", "eurocargo", "s-way"),
    "man": ("tge", "tgl", "tgm", "tgs", "tgx"),
    "mitsubishi": ("canter", "fuso", "l200"), "fuso": ("canter",),
    "toyota": ("hilux", "proace", "hiace"), "nissan": ("navara", "interstar", "primastar", "nv200", "nv300", "nv400"),
    "isuzu": ("d-max", "n-series"), "hyundai": ("h1", "h-1", "h100", "h-100"),
}


def summary_context(title, year):
    text = clean(title)
    brand, remaining = "unknown", ""
    for prefix in sorted(BRANDS, key=lambda s: (-len(s), s)):
        match = re.match(re.escape(prefix) + r"(?:\s+|$)", text)
        if match:
            brand, remaining = BRANDS[prefix], text[match.end():]
            break
    family = "unknown"
    for token in sorted(FAMILIES.get(brand, ()), key=lambda s: (-len(s), s)):
        if re.match(re.escape(token) + r"(?:\s|$)", remaining):
            family = token
            break
    return {"brand": brand, "family": family, "year": float(year)}


def build(archive):
    sheets = workbook_sheets(archive)
    headers = {m: recognized_header(archive, member, m) for m, member in sheets}
    needed = set()
    for month, member in sheets:
        for _, record in rows(archive, member, headers[month], Counter()):
            needed.update(v for kind, v in record.values() if kind == "shared")
    strings = resolve_shared_strings(archive, needed)
    seen, records, counts = set(), [], Counter()
    for month, member in sheets:
        role = next((role for role, months in ROLE_MONTHS.items() if month in months), None)
        fields = {HEADER_FIELDS[v] for v in headers[month].values()} - {"price", "additional"}
        for coordinate, raw in rows(archive, member, headers[month], Counter()):
            values = {f: (strings[raw[f][1]] if raw[f][0] == "shared" else raw[f][1]) if f in raw else "" for f in fields}
            if role is None or not eligible(values):
                continue
            group = fingerprint(values, ("name", "year")).hex()
            if group in seen:
                counts["repeated_eligible_title_year_records_dropped"] += 1
                continue
            seen.add(group)
            kw = number(values.get("power_kw", ""), "power_kw")
            if kw is None or kw <= 0:
                kw = number(values.get("power_ps", ""), "power_ps") * Decimal("0.73549875")
            context = summary_context(values["name"], number(values["year"], "year"))
            record = {
                "record_key": f"AT2023-{month}-{coordinate:06}", "month": month, "source_row": coordinate,
                "group_hash": group, "role": role,
                "oof_fold": int(hashlib.sha256(("at2023-oof-v1/" + group).encode()).hexdigest()[:16], 16) % 5 if role == "development" else None,
                "context": context,
                "actions": {"mileage_km": float(number(values["mileage_km"], "mileage_km")),
                            "power_kw": float(kw), "fuel": classification(values["fuel"], "fuel"),
                            "transmission": classification(values["transmission"], "transmission")},
            }
            records.append(record)
            counts[role] += 1
            counts[role + "_unknown_brand"] += int(context["brand"] == "unknown")
            counts[role + "_unknown_family"] += int(context["family"] == "unknown")
    if len(records) != len({r['group_hash'] for r in records}):
        raise ValueError("One representative per title-year group required")
    return records, dict(counts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("Fresh output directory required")
    probe(args.source)
    with zipfile.ZipFile(args.source) as z:
        records, counts = build(z)
    args.output_dir.mkdir(parents=True, mode=0o700)
    path = args.output_dir / "private_price_free_cohort.json"
    path.write_text(json.dumps(records, separators=(',', ':')) + '\n', encoding='utf-8')
    path.chmod(0o600)
    receipt = {
        "stage": "PRICE_FREE_COHORT_PREPARED_NOT_EXPERIMENT_GO", "source_sha256": AT_SHA256,
        "cohort_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "counts": counts,
        "role_months": ROLE_MONTHS, "actions": ACTIONS,
        "one_earliest_eligible_record_per_title_year_group": True,
        "title_year_groups_are_not_verified_vehicle_ids": True,
        "individual_records_private_not_publicly_distributed": True,
        "prices_decoded": False, "model_fit_or_test_evaluation_started": False,
    }
    (args.output_dir / 'cohort_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
