"""Strict train/validation prediction pairs for choosing a fixed field.

Training action outcomes come from cross-fitted *train-only* base models.
Validation prediction pairs come from a base model fitted only on train.
Validation labels never train the base or risk model used to rank validation.
No calibration or sealed-test file is accepted by this program.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from generate_bvival_prediction_pairs import (
    ACTION_FIELDS,
    crossfit_development,
    file_sha256,
    fit_ensemble,
    infer_feature_columns,
    predict_listing_actions,
    write_pair_bundle,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--action-fields", default=",".join(ACTION_FIELDS))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", default="13,42,2026")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--l2-leaf-reg", type=float, default=10.0)
    args = parser.parse_args()
    action_fields = tuple(value.strip() for value in args.action_fields.split(",") if value.strip())
    if not action_fields or len(action_fields) != len(set(action_fields)):
        raise ValueError("Action fields must be nonempty and unique")
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Seeds must be nonempty and unique")
    train = pd.read_csv(args.train)
    validation = pd.read_csv(args.validation)
    if set(train["listing_id"]).intersection(validation["listing_id"]):
        raise ValueError("Train and validation listing IDs overlap")
    for name, frame in (("train", train), ("validation", validation)):
        if "target_log" not in frame or frame["listing_id"].duplicated().any():
            raise ValueError(f"{name} requires unique IDs and target_log")
    feature_columns, categorical_columns = infer_feature_columns(
        train, action_fields=action_fields
    )
    valid_features, valid_categorical = infer_feature_columns(
        validation, action_fields=action_fields
    )
    if (feature_columns, categorical_columns) != (valid_features, valid_categorical):
        raise ValueError("Train/validation feature schemas differ")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_pairs, train_pre = crossfit_development(
        train,
        feature_columns=feature_columns,
        categorical_columns=categorical_columns,
        folds=args.folds,
        seeds=seeds,
        iterations=args.iterations,
        depth=args.depth,
        learning_rate=args.learning_rate,
        l2_leaf_reg=args.l2_leaf_reg,
        action_fields=action_fields,
    )
    train_hashes = write_pair_bundle(
        name="train_oof", pairs=train_pairs, pre_features=train_pre,
        labels=train, output_dir=args.output_dir,
    )
    train_models = fit_ensemble(
        train,
        feature_columns=feature_columns,
        categorical_columns=categorical_columns,
        seeds=seeds,
        iterations=args.iterations,
        depth=args.depth,
        learning_rate=args.learning_rate,
        l2_leaf_reg=args.l2_leaf_reg,
        action_fields=action_fields,
    )
    validation_pairs, validation_pre = predict_listing_actions(
        train_models, validation,
        feature_columns=feature_columns,
        categorical_columns=categorical_columns,
        fold_id=-2,
        prediction_is_oof=True,  # strictly out of the train-only base fit
        action_fields=action_fields,
    )
    validation_hashes = write_pair_bundle(
        name="validation_heldout", pairs=validation_pairs,
        pre_features=validation_pre, labels=validation,
        output_dir=args.output_dir,
    )
    models = []
    for seed, model in zip(seeds, train_models):
        path = args.output_dir / f"train_only_base_seed_{seed}.cbm"
        model.save_model(path)
        models.append({"seed": seed, "sha256": file_sha256(path)})
    audit = {
        "protocol": "strict-train-validation-fixed-field-selection",
        "train_listing_count": int(train["listing_id"].nunique()),
        "validation_listing_count": int(validation["listing_id"].nunique()),
        "train_validation_id_overlap": 0,
        "train_pair_count": int(len(train_pairs)),
        "validation_pair_count": int(len(validation_pairs)),
        "action_fields": list(action_fields),
        "feature_columns": feature_columns,
        "categorical_columns": categorical_columns,
        "seeds": seeds,
        "folds": args.folds,
        "model_parameters": {
            "iterations": args.iterations,
            "depth": args.depth,
            "learning_rate": args.learning_rate,
            "l2_leaf_reg": args.l2_leaf_reg,
        },
        "inputs": {"train_sha256": file_sha256(args.train),
                   "validation_sha256": file_sha256(args.validation)},
        "outputs": {"train_oof": train_hashes,
                    "validation_heldout": validation_hashes},
        "train_only_base_models": models,
        "validation_labels_used_in_base_fit": False,
        "sealed_test_input_read": False,
        "performance_metrics_computed": False,
    }
    (args.output_dir / "field_selection_pair_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
