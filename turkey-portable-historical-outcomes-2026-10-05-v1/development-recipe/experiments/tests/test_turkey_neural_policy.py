import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from test_turkey_policy_components import fixture
from turkey_neural_policy import fit_network, negative_expected_gain, run_neural_policy


def test_expected_gain_gradient_rewards_good_and_penalizes_harmful_action():
    logits = torch.zeros((1, 3), requires_grad=True)
    loss = negative_expected_gain(logits, torch.tensor([[20., -5., 0.]]), 100.)
    loss.backward()
    assert logits.grad[0, 0] < 0 and logits.grad[0, 1] > 0
    assert loss.item() == pytest.approx(-.05)


def test_train_only_scale_preserves_global_absolute_gain_weighting():
    logits = torch.zeros((2, 3), requires_grad=True)
    gains = torch.tensor([[30., 0., 0.], [300., 0., 0.]])
    negative_expected_gain(logits, gains, 100.).backward()
    assert (logits.grad[1] / logits.grad[0]).tolist() == pytest.approx([10., 10., 10.])


def test_listing_constant_subtraction_preserves_expected_loss_gradient():
    first, second = torch.zeros((2, 3), requires_grad=True), torch.zeros((2, 3), requires_grad=True)
    gains = torch.tensor([[30., -10., 0.], [15., 5., -5.]])
    before = torch.tensor([[100.], [200.]])
    negative_expected_gain(first, gains, 100.).backward()
    (torch.softmax(second, dim=1) * (before - gains) / 100.).sum(dim=1).mean().backward()
    assert torch.allclose(first.grad, second.grad, atol=1e-7)


def test_actual_neural_selection_and_fixed_epoch_final_fits(fixture, tmp_path):
    train, valid, plan, config = fixture
    plan["neural_comparator"].update(maximum_epochs=3, patience=2)
    result = run_neural_policy(train, valid, plan, config, valid["keys"][:2], output_directory=tmp_path)
    assert result["network_fit_count"] == 6 and len(result["candidate_trials"]) == 3
    assert not result["official_method_reproduction_completed"]
    assert np.shape(result["probability"]) == (20, 3)
    assert np.allclose(np.asarray(result["probability"]).sum(axis=1), 1)
    assert all(not a["validation_labels_used"] and a["epochs_run"] == result["selected_epoch"] for a in result["final_fit_audit"])
    assert len(list(tmp_path.glob("*.pt"))) == 6


@pytest.mark.parametrize("mutation", ["cohort_short", "cohort_unknown", "cohort_duplicate", "hidden_value", "not_oof"])
def test_neural_boundary_rejects_invalid_roles_before_training(fixture, mutation):
    train, valid, plan, config = fixture
    cohort = valid["keys"][:2]
    if mutation == "cohort_short":
        cohort = cohort[:1]
    elif mutation == "cohort_unknown":
        cohort[0] = "test_record"
    elif mutation == "cohort_duplicate":
        cohort[1] = cohort[0]
    elif mutation == "hidden_value":
        train["contexts"][0]["kilometre"] = "10000"
    else:
        train["target_origin"] = "in_sample"
    with pytest.raises(ValueError):
        run_neural_policy(train, valid, plan, config, cohort)


def test_fixed_epoch_training_cannot_use_validation_labels(fixture):
    train, valid, plan, config = fixture
    settings = plan["neural_comparator"]
    x, v = np.ones((8, 2), dtype=np.float32), np.ones((3, 2), dtype=np.float32)
    gains = np.repeat([[10., -5., 0.]], 8, axis=0)
    _, first, audit = fit_network(x, gains, v, settings, seed=13, learning_rate=.001, epochs=2, scale=100)
    _, second, _ = fit_network(x, gains, v, settings, seed=13, learning_rate=.001, epochs=2, scale=100)
    assert np.array_equal(first, second) and not audit["validation_labels_used"]


def test_neural_preprocessing_is_fitted_on_train_only(fixture):
    from turkey_neural_policy import make_preprocessor
    from turkey_policy_components import context_frame
    train, valid, plan, config = fixture
    valid["contexts"][0]["marka"] = "VALIDATION_ONLY_BRAND"
    prep = make_preprocessor(plan, config)
    prep.fit(context_frame(train["contexts"], plan, config)[0])
    encoded = prep.transform(context_frame(valid["contexts"], plan, config)[0])
    assert np.isfinite(encoded).all()
    assert all("VALIDATION_ONLY_BRAND" not in c for c in prep.named_transformers_["categorical"].categories_)


def test_learning_rate_tie_uses_smallest_rate_and_final_epochs_are_frozen(fixture, monkeypatch):
    import turkey_neural_policy as module
    train, valid, plan, config = fixture
    calls = []
    def fake_fit(x, gain, v, settings, *, seed, learning_rate, epochs, scale, validation=None, cohort=None):
        calls.append((seed, learning_rate, epochs, validation is not None))
        return None, np.repeat([[.1, .2, .7]], len(v), axis=0), {
            "seed": seed, "learning_rate": learning_rate, "selected_epoch": 4 if validation is not None else epochs,
            "epochs_run": epochs, "validation_labels_used": validation is not None,
            "validation_selection_post_mae": 100 if validation is not None else None}
    monkeypatch.setattr(module, "fit_network", fake_fit)
    result = run_neural_policy(train, valid, plan, config, valid["keys"][:2])
    assert result["selected_learning_rate"] == .0003 and result["selected_epoch"] == 4
    assert calls[3:] == [(seed, .0003, 4, False) for seed in plan["seeds"]]


def test_neural_bridge_refuses_truthy_string_validity(fixture):
    from turkey_neural_policy import restore_material
    train, _, _, _ = fixture
    train["valid_mask"] = ["false"] * len(train["keys"])
    with pytest.raises(ValueError, match="JSON booleans"):
        restore_material(train)
