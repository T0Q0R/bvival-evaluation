"""Separate neural environment: saved-network inference on initial context only."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import torch

from turkey_development_io import sha256
from turkey_neural_policy import make_network, probabilities
from turkey_policy_components import context_frame
from turkey_price_free_scoring import verify_completed_package


def score_neural(task, directory, *, project_root):
    if (set(task) != {"stage", "contexts", "plan", "config", "development_audit_sha256"}
            or task["stage"] != "INITIAL_ONLY_INFERENCE_NO_LABEL_RELEASE"):
        raise ValueError("Exact initial-only neural task required; supervision forbidden")
    # Context checks before reading/loading models; reject hidden/after/outcome roles.
    frame, _ = context_frame(task["contexts"], task["plan"], task["config"])
    audit, plan, config = verify_completed_package(directory, project_root=project_root)
    if (task["development_audit_sha256"] != sha256(directory / "development_execution_audit.json")
            or task["plan"] != plan or task["config"] != config):
        raise ValueError("Neural task does not match verified saved-model package")
    expected = audit["neural_runtime_environment"]
    observed = {"python": platform.python_version(),
                "packages": {p: importlib.metadata.version(p) for p in expected["packages"]}}
    if "platform" in expected:
        observed["platform"] = platform.platform()
    if observed != expected:
        raise ValueError("Saved neural runtime changed")
    prep = joblib.load(directory / "neural/train_fitted_preprocessing.joblib")
    matrix = np.asarray(prep.transform(frame), dtype=np.float32)
    if not np.isfinite(matrix).all():
        raise ValueError("Finite train-fitted neural inference matrix required")
    torch.set_num_threads(4)
    per_seed = {}
    for seed in plan["seeds"]:
        network = make_network(matrix.shape[1], plan["neural_comparator"])
        network.load_state_dict(torch.load(directory / "neural" / f"final_seed_{seed}.pt", weights_only=True))
        per_seed[str(seed)] = probabilities(network, matrix).tolist()
    return {"probability": np.mean(list(per_seed.values()), axis=0).tolist(), "per_seed_probability": per_seed,
            "neural_models_reloaded": 3, "refits": 0, "reselection": False, "input_contains_supervision": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--development-run", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(score_neural(json.loads(args.task.read_text()), args.development_run,
                                 project_root=Path(__file__).resolve().parents[1]), allow_nan=False))


if __name__ == "__main__":
    main()
