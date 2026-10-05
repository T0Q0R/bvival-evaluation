"""Read-only reload of locally produced neural policies; no refitting/selection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch

from turkey_neural_policy import make_network, probabilities
from turkey_policy_components import context_frame


def replay(task_path, model_directory):
    # Caller verifies the saved package hashes before any deserialization.
    task = json.loads(task_path.read_text())
    audit = json.loads((model_directory / "neural_audit.json").read_text())
    prep = joblib.load(model_directory / "train_fitted_preprocessing.joblib")
    frame, _ = context_frame(task["validation"]["contexts"], task["plan"], task["config"])
    matrix = np.asarray(prep.transform(frame), dtype=np.float32)
    torch.set_num_threads(4)
    per_seed = {}
    for seed in task["plan"]["seeds"]:
        network = make_network(matrix.shape[1], task["plan"]["neural_comparator"])
        network.load_state_dict(torch.load(model_directory / f"final_seed_{seed}.pt", weights_only=True))
        actual = probabilities(network, matrix)
        if not np.array_equal(actual, np.asarray(audit["per_seed_probability"][str(seed)])):
            raise ValueError("Saved neural probability does not match reloaded model")
        per_seed[str(seed)] = actual.tolist()
    mean = np.mean(list(per_seed.values()), axis=0)
    if not np.array_equal(mean, np.asarray(audit["probability"])):
        raise ValueError("Saved neural ensemble mismatch")
    return {"neural_models_reloaded": 3, "refits": 0, "probability": mean.tolist(), "per_seed_probability": per_seed}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(replay(args.task, args.model_dir)))


if __name__ == "__main__":
    main()
