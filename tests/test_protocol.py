import itertools

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

from financial_supervised_ad.data import prepare_features, split_indices, synthetic_data, validate_frame
from financial_supervised_ad.experiment import preprocessing, run_experiment, score_model
from financial_supervised_ad.metrics import evaluate, select_threshold
from financial_supervised_ad.models import METHODS


def test_threshold_matches_exhaustive_search_with_ties():
    scores = np.array([0.1, 0.4, 0.4, 0.8, 0.9, 0.9])
    for labels in itertools.product([0, 1], repeat=len(scores)):
        y = np.array(labels)
        if len(set(y)) != 2:
            continue
        candidates = np.r_[np.nextafter(scores.max(), np.inf), np.unique(scores)[::-1]]
        expected = max(candidates, key=lambda t: f1_score(y, scores >= t, zero_division=0))
        assert select_threshold(y, scores) == expected


def test_fpr_threshold_is_validation_constrained():
    y = np.array([0, 0, 0, 1, 1])
    scores = np.array([0.9, 0.2, 0.1, 0.9, 0.8])
    chosen = select_threshold(y, scores, "fpr", max_fpr=0)
    assert not (scores >= chosen).any()  # tied normal/fraud at top: predict-none
    chosen = select_threshold(y, scores, "fpr", max_fpr=1/3)
    assert evaluate(y, scores, chosen)["recall"] == 1
    assert evaluate(y, scores, chosen)["fpr"] == 1/3


def test_temporal_ties_do_not_cross_boundaries():
    frame = pd.DataFrame({"Time": np.repeat(np.arange(10), 4), "Class": [0, 1]*20})
    groups = split_indices(frame)
    assert len(set(np.concatenate(list(groups.values())))) == len(frame)
    for first, second in (("train", "validation"), ("validation", "test")):
        assert frame.iloc[groups[first]].Time.max() < frame.iloc[groups[second]].Time.min()


def test_train_only_preprocessing_and_unknown_category():
    train = pd.DataFrame({"amount": [1.0, 3.0, np.nan], "merchant": ["a", "b", "a"]})
    transform = preprocessing(train).fit(train)
    assert transform.named_transformers_["numeric"].named_steps["impute"].statistics_[0] == 2
    unseen = pd.DataFrame({"amount": [1000000.0], "merchant": ["unseen"]})
    assert np.isfinite(transform.transform(unseen)).all()
    assert transform.named_transformers_["numeric"].named_steps["impute"].statistics_[0] == 2


def test_invalid_labels_and_temporal_one_class_fail():
    with pytest.raises(ValueError):
        validate_frame(pd.DataFrame({"Class": [0, None, 1]}), "Class")
    with pytest.raises(ValueError):
        split_indices(pd.DataFrame({"Time": range(30), "Class": [0]*28+[1]*2}))


def test_all_ten_methods_train_save_and_reproduce_scores(tmp_path):
    frame = synthetic_data(1000, 5)
    output = tmp_path / "run"
    results = run_experiment(frame, output, split="stratified", seed=5,
                             quick=True, balance="balanced", save_models=True)
    assert set(results.method) == set(METHODS)
    for row in results.to_dict("records"):
        assert 0 <= row["average_precision"] <= 1
        assert row["tn"]+row["fp"]+row["fn"]+row["tp"] == 200
        bundle = joblib.load(output / f"{row['method']}.joblib")
        saved = pd.read_csv(output / f"{row['method']}_predictions.csv.gz")
        original = prepare_features(frame, "Class", "Time")
        reloaded = score_model(bundle["pipeline"], original.iloc[saved.row_id])
        np.testing.assert_allclose(saved.score, reloaded, atol=1e-12)
        assert np.array_equal(saved.prediction, (reloaded >= bundle["threshold"]).astype(int))
    with pytest.raises(FileExistsError):
        run_experiment(frame, output, split="stratified")


def test_metrics_have_known_values():
    result = evaluate([0, 0, 1, 1], [0.1, 0.7, 0.6, 0.9], 0.5)
    assert result["tp"] == 2 and result["fp"] == 1
    assert result["recall"] == 1 and result["fpr"] == 0.5
    assert result["roc_auc"] == 0.75
