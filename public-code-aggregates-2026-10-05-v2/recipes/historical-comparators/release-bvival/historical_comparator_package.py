"""Build/check the bounded fixed-benefit and neighbor reconstruction companion.

Only code, config templates and six aggregate references are copied. Primary
rows are obtained separately by executing the already verified source recipes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_source_bundle import safe_file, source_closure, MARKERS
from historical_reconstruction_contract import LOCK, LOCK_SOURCE, LOCK_SHA, sha256

VERSION = "historical-core-comparator-reconstruction-2026-10-04-v1"
PRIMARY_MANIFEST_SHA = "fa2cef5b9009a2984db59548265c8f62dcbd7e6bba65f6f64e4ad06eb438d551"
FIXED_CONFIG_SOURCE = "experiments/configs/bvival_historical_fixed_benefit_2026-10-04.json"
FIXED_CONFIG_ORIGINAL_SHA = "77744a3bd46ccf7becb960af8f7c267d0e7da47b5301d7dd5a0e02534649cb87"
FIXED_TEMPLATE = "recipes/fixed_benefit.template.json"
FIXED_TEMPLATE_SHA = "5a24c1868bd309d37c657f82a22252a713f8b4818756317d974cabc66cddc728"
RUNTIME_TOKEN = "__VALIDATED_RUNTIME_PYTHON__"
NEIGHBOR_CONFIG = "experiments/configs/bvival_neighbor_historical_v1_2026-10-01.json"
NEIGHBOR_CONFIG_SHA = "043a507ebdf9e0276b2787a64aa7f5ca729cb9c8b857180c052894f3289faa14"
FROZEN_SOURCE_SHAS = {
    "experiments/run_historical_fixed_benefit_control.py": "d1a4cfbe41034bbb3b023a3be9d7dcca1cfe775820e3cbbda5ade4c95022c32d",
    "experiments/check_historical_fixed_benefit_reconstruction.py": "af4732f562eba2bdc673a347fd78da0c564a2e553b5949de78fba62683d3c15b",
    "experiments/run_bvival_neighbor_historical.py": "289d056c4de20b61920f00b79e35a59f97803c7c1eaade41d49b0d195a113de9",
    "experiments/bvival_neighbor_available.py": "389b90d5243cba355998a078992df5d64580c281f2ade4e1e2688240dff2b84b",
    "experiments/bvival_neighbor_loss.py": "42d2131638b4b8bdaacf2ac3a0b6d6f5582fd7800e1455e913ddba4b03880959",
}
REFERENCE_SHAS = {
    "references/fixed/mucars.json": "41fee5d6762f57b31346214ea8c4e3b48af17c4c8f2d76cc735263b2a980f48d",
    "references/fixed/autoscout24.json": "f53b91759d3a8e375b8cdc2e0a05785d29d357d973d6839025b17d617c7faa85",
    "references/fixed/jucars.json": "e8e622fd2752646f921d981ae268d246a9a1575c2f481a894e9d4ec4b19dcb54",
    "references/neighbor/MUCars.json": "8117b8eaa519f189689317429d559ae9d3f1c0d455a2fea5497db039a05fbd33",
    "references/neighbor/JUCars.json": "316faedcd123402fb509e722ffcbe37b2e11d000ce8dfe77f32bf5666a574b5c",
    "references/neighbor/AutoScout24.json": "bc1592120e2e0bda5727be80e9c6e3b9ab6f594d529a8de89915ebb383b1cdae",
}
NEIGHBOR_ROW_HASHES = {
    "MUCars": {"scores.csv": "9be935055e8207affd0ec004c64eeadd71894da2eeea89e251e6c871f8844166",
               "allocations.json": "2616d4786c7eb330e49f462e08bf103ad24fdc61d44546b7ac2945a035268e16"},
    "JUCars": {"scores.csv": "bc0b2b1769e24d432d387084ea0af57a21bc3f98b5703ac74750a8c45f47c8f4",
               "allocations.json": "14c067c5378cbbce569a7d4c0b58d2c105664f7d8d830fb35e58a0f9783999a7"},
    "AutoScout24": {"scores.csv": "0621911cda4702c2d978ef7328891135f666954f45e9b2c5e5d5b0bdf6bdeded",
                   "allocations.json": "abfc792320f72ee084c5cac25a47766d263954d6b2c38156eafd579ca96d9850"},
}
ROOTS = sorted(set(FROZEN_SOURCE_SHAS) | {
    "release-bvival/historical_comparator_package.py",
    "release-bvival/run_historical_comparator_reconstruction.py",
    "experiments/tests/test_bvival_neighbor_historical.py",
})
FALSE_FLAGS = ("publicly_archived", "project_license_selected", "raw_or_row_level_data_included",
               "new_independent_confirmation", "full_empirical_pipeline_portable")


def verify(root):
    root = Path(root).resolve(strict=True)
    manifest = json.loads(safe_file(root, "manifest.json").read_text())
    if manifest.get("version") != VERSION or any(manifest.get(k) is not False for k in FALSE_FLAGS):
        raise ValueError("Comparator package/version/capability boundary differs")
    if (manifest.get("public_archive_or_reviewer_link") is not None
            or manifest.get("primary_package_manifest_sha256") != PRIMARY_MANIFEST_SHA
            or manifest.get("historical_tests_already_opened") is not True
            or manifest.get("original_fixed_config_sha256") != FIXED_CONFIG_ORIGINAL_SHA
            or manifest.get("fixed_template_only_runtime_pointer_changed") is not True
            or manifest.get("neighbor_row_reference_hashes") != NEIGHBOR_ROW_HASHES):
        raise ValueError("Access/parent/row-reference contract differs")
    closure = source_closure(root, ROOTS)
    if closure != manifest["static_import_closure"] or closure["unresolved_dynamic_imports"]:
        raise ValueError("Static source closure differs")
    if set(closure["external_import_roots"]) - {"numpy", "pandas", "scipy", "sklearn", "catboost", "pytest"}:
        raise ValueError("Unexpected dependency")
    pinned = {**FROZEN_SOURCE_SHAS, **REFERENCE_SHAS, FIXED_TEMPLATE: FIXED_TEMPLATE_SHA,
              NEIGHBOR_CONFIG: NEIGHBOR_CONFIG_SHA, LOCK: LOCK_SHA}
    expected = set(closure["local_source_sha256"]) | set(pinned) | {"README.md"}
    records = {}
    for record in manifest["files"]:
        if record["file"] in records:
            raise ValueError("Duplicate payload record")
        records[record["file"]] = record
    if set(records) != expected:
        raise ValueError("Exact comparator payload allowlist differs")
    actual = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("Symlink package content")
        if path.is_file():
            actual.add(str(path.relative_to(root)))
    if actual != expected | {"manifest.json"}:
        raise ValueError("Unexpected/missing package content")
    for relative, record in records.items():
        path = safe_file(root, relative)
        data = path.read_bytes()
        if len(data) != record["bytes"] or sha256(path) != record["sha256"]:
            raise ValueError("Payload hash/size differs")
        if relative in pinned and sha256(path) != pinned[relative]:
            raise ValueError("Fixed comparator/config/reference drift")
        if any(marker in data for marker in MARKERS):
            raise ValueError("Private content marker")
    return {"stage": "COMPARATOR_RECIPE_PACKAGE_VERIFIED_NOT_EMPIRICAL_RECONSTRUCTION",
            "payload_files": len(records), "aggregate_references": 6,
            "source_rows_or_models_read": False, "statistical_analysis_called": False}


def build(root, destination):
    root, destination = Path(root).resolve(strict=True), Path(destination).absolute()
    if destination.exists() or destination.is_symlink() or any(p.is_symlink() for p in destination.parents):
        raise ValueError("New non-symlink destination required")
    if any(destination.is_relative_to(root / d) for d in ("experiments", "paper-bvival", "paper-journal")):
        raise ValueError("Do not write inside scientific source/data trees")
    original = safe_file(root, FIXED_CONFIG_SOURCE)
    if sha256(original) != FIXED_CONFIG_ORIGINAL_SHA:
        raise ValueError("Original fixed comparison config changed")
    fixed = json.loads(original.read_text())
    fixed["runtime_python"] = RUNTIME_TOKEN
    template = (json.dumps(fixed, indent=2, sort_keys=True) + "\n").encode()
    if hashlib.sha256(template).hexdigest() != FIXED_TEMPLATE_SHA:
        raise ValueError("Only the runtime pointer may be replaced in the template")
    closure = source_closure(root, ROOTS)
    mapping = {p: p for p in closure["local_source_sha256"]}
    mapping.update({LOCK: LOCK_SOURCE, "README.md": "release-bvival/HISTORICAL_COMPARATOR_RECONSTRUCTION.md",
                    NEIGHBOR_CONFIG: NEIGHBOR_CONFIG})
    for target in REFERENCE_SHAS:
        group, name = target.split("/")[1:]
        source = ("experiments/outputs/bvival_fixed_benefit_control_2026-10-04_v1/" + name[:-5] + "/aggregate_result.json"
                  if group == "fixed" else "experiments/outputs/bvival_neighbor_historical_2026-10-01_r3/" + name[:-5] + "/results.json")
        mapping[target] = source
    payload = {target: safe_file(root, source).read_bytes() for target, source in mapping.items()}
    payload[FIXED_TEMPLATE] = template
    pinned = {**FROZEN_SOURCE_SHAS, **REFERENCE_SHAS, FIXED_TEMPLATE: FIXED_TEMPLATE_SHA,
              NEIGHBOR_CONFIG: NEIGHBOR_CONFIG_SHA, LOCK: LOCK_SHA}
    records = []
    for relative, data in sorted(payload.items()):
        digest = hashlib.sha256(data).hexdigest()
        if any(marker in data for marker in MARKERS) or relative in pinned and digest != pinned[relative]:
            raise ValueError("Unreviewed/changed comparator payload: " + relative)
        records.append({"file": relative, "bytes": len(data), "sha256": digest})
    manifest = {"version": VERSION, **{k: False for k in FALSE_FLAGS},
                "public_archive_or_reviewer_link": None, "historical_tests_already_opened": True,
                "primary_package_manifest_sha256": PRIMARY_MANIFEST_SHA,
                "neighbor_row_reference_hashes": NEIGHBOR_ROW_HASHES,
                "original_fixed_config_sha256": FIXED_CONFIG_ORIGINAL_SHA,
                "fixed_template_only_runtime_pointer_changed": True,
                "static_import_closure": closure, "files": records}
    destination.mkdir(parents=True, mode=0o700, exist_ok=False)
    for relative, data in payload.items():
        path = destination / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(data)
    with (destination / "manifest.json").open("x") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    return verify(destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path)
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.package_root and not args.project_root and not args.output_dir:
        result = verify(args.package_root)
    elif args.project_root and args.output_dir and not args.package_root:
        result = build(args.project_root, args.output_dir)
    else:
        parser.error("Check one package OR create a new package from the project")
    print(json.dumps(result, indent=2))
