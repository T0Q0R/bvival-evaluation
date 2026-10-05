"""Strict pre-action and one-reveal interfaces for the new source.

Generic English outcome-token checks do not forbid raw Turkish acquisition
fields. Future BVI-Val reuse must use these source-specific role allowlists.
This is a data-access guard, not a new learning algorithm or truth guarantee.
"""
from __future__ import annotations

import math

from build_turkey_price_free_manifest import validate_config


def initial_context(values: dict, config: dict) -> dict:
    validate_config(config)
    if set(values) != set(config["initial_visible_fields"]):
        raise ValueError("Initial context must contain only the exact pre-action field allowlist")
    return {field: values[field] for field in config["initial_visible_fields"]}


def valuation_state(values: dict, config: dict, *, revealed_action: str | None = None,
                    revealed_value: str | None = None) -> dict:
    state = initial_context(values, config)
    actions = config["action_fields"]
    if revealed_action is None:
        if revealed_value is not None:
            raise ValueError("A field value requires an explicitly selected action")
    elif revealed_action not in actions or not isinstance(revealed_value, str) or not revealed_value:
        raise ValueError("Exactly one known action with a nonempty scalar value is required")
    for field in actions:
        state[field] = revealed_value if field == revealed_action else ""
        state["hidden_" + field] = int(field != revealed_action)
    return state


def policy_context(values: dict, config: dict, *, before_prediction_log: float,
                   before_disagreement_log: float, action_id: str | None = None) -> dict:
    context = initial_context(values, config)
    if (not math.isfinite(before_prediction_log) or not math.isfinite(before_disagreement_log)
            or before_disagreement_log < 0):
        raise ValueError("Before-action prediction summaries must be finite; disagreement nonnegative")
    context.update({"before_prediction_log": before_prediction_log,
                    "before_disagreement_log": before_disagreement_log})
    if action_id is not None:
        if action_id not in config["action_fields"]:
            raise ValueError("Unknown candidate action")
        context["action_id"] = action_id
    return context
