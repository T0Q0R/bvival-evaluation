import copy
import json
from pathlib import Path

from run_turkey_synthetic_integration import reduced_synthetic_plan, synthetic_dataset, synthetic_price_check
from train_turkey_nested_valuation import validate_dataset
from turkey_execution_contract import validate_plan


def test_smoke_reduction_never_changes_original_scientific_grid():
    directory = Path(__file__).parents[1] / "configs"
    plan = json.loads((directory / "turkey_execution_plan_v1_2026-09-30.json").read_text())
    cohort = json.loads((directory / "turkey_price_free_cohort_2026-09-30.json").read_text())
    original = copy.deepcopy(plan)
    reduced = reduced_synthetic_plan(plan)
    assert plan == original
    assert reduced["evidence_status"] == "synthetic_smoke_only_not_source_development"
    assert reduced["valuation"]["candidates"][0]["params"]["iterations"] == 5
    assert original["valuation"]["candidates"][0]["params"]["iterations"] == 400
    validate_plan(reduced, cohort)
    validate_dataset(synthetic_dataset(cohort), reduced, cohort)


def test_synthetic_workbook_probe_never_parses_unselected_price():
    access = synthetic_price_check()
    assert access["selected_valid_targets"] == 2
    assert access["unselected_price_cells_skipped_before_xml_parser"] == 1
    assert access["unselected_price_values_decoded"] == 0
