import itertools
import importlib.util

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(importlib.util.find_spec("torch") is None or
    importlib.util.find_spec("interpret") is None, reason="Install optional recent2026 dependencies")


def test_orthogonal_arrays_have_balanced_pairwise_levels():
    from financial_supervised_ad.recent2026.ebm import orthogonal_array
    for prime, columns in ((3, 4), (5, 5)):
        array = orthogonal_array(prime, columns)
        for left, right in itertools.combinations(range(columns), 2):
            assert len({tuple(row) for row in array[:, [left, right]]}) == prime**2


def test_windows_endpoints_and_split_isolation():
    from financial_supervised_ad.recent2026.sttn import make_windows
    features = np.arange(60).reshape(20, 3)
    labels, ids = np.arange(20) % 2, np.arange(20)
    for group in (slice(0, 10), slice(10, 20)):
        windows, y, end = make_windows(features[group], labels[group], ids[group], 4)
        for window, label, endpoint in zip(windows, y, end):
            np.testing.assert_array_equal(window[-1], features[endpoint])
            assert label == labels[endpoint]
            assert window[0, 0] >= features[group][0, 0]


def test_infonce_matches_independent_logsumexp_formula_and_has_gradients():
    import torch
    from financial_supervised_ad.recent2026.sttn_architecture import infonce_loss
    torch.manual_seed(1)
    first, second = torch.randn(4, 8, requires_grad=True), torch.randn(4, 8, requires_grad=True)
    actual = infonce_loss(first, second, temperature=0.2)
    reps = torch.nn.functional.normalize(torch.cat([first, second]), dim=1)
    similarities = reps @ reps.T / 0.2
    values = []
    for i in range(8):
        other = torch.cat([similarities[i, :i], similarities[i, i+1:]])
        values.append(-similarities[i, (i+4) % 8] + torch.logsumexp(other, dim=0))
    torch.testing.assert_close(actual, torch.stack(values).mean())
    actual.backward()
    assert torch.isfinite(first.grad).all() and first.grad.abs().sum() > 0


def test_both_2026_methods_train_export_and_audit(tmp_path):
    from financial_supervised_ad.recent2026.cli import main
    import importlib.util
    from pathlib import Path
    audit_path = Path(__file__).parents[1] / "scripts/audit_results.py"
    spec = importlib.util.spec_from_file_location("audit_2026", audit_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for method in ("optimized_ebm_2026", "sttn_cp_2026"):
        output = tmp_path / method
        main(["--method", method, "--quick", "--n-samples", "1000", "--output", str(output), "--device", "cpu"])
        module.audit(output)
        assert (output / "inference.json").exists()
        from financial_supervised_ad.data import synthetic_data
        from financial_supervised_ad.recent2026.predict import predict
        import pandas as pd
        frame = synthetic_data(1000, 42)
        frame["Time"] = np.arange(1000)
        reloaded = predict(output, frame)
        exported = pd.read_csv(output / f"{method}_predictions.csv.gz", float_precision="round_trip")
        np.testing.assert_allclose(reloaded.loc[exported.row_id, "score"], exported.score,
                                   atol=1e-6, rtol=1e-5)
        if method == "sttn_cp_2026":
            import torch
            from financial_supervised_ad.recent2026.sttn_architecture import STTNCP
            saved = torch.load(output / "best_model.pt", map_location="cpu", weights_only=True)
            model = STTNCP(**saved["model_config"])
            model.load_state_dict(saved["state_dict"])
            assert model(torch.zeros(2, 4, 20)).shape == (2, 2)
