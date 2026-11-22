import importlib.util
from pathlib import Path

import pandas as pd
import pytest

MODULE_PATH = Path(__file__).parents[1] / "scripts/v0.5/plot_nemo_figure2_partial.py"
SPEC = importlib.util.spec_from_file_location("nemo_fig2_partial", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _rows():
    return pd.DataFrame([
        {"gene": f"AT{i}", "family": f"F{i}", "chrom": "chr1", "split": "test",
         "observed_log10_tpm": float(i), "predicted_log10_tpm": float(i) + 0.25,
         "model": MODULE.MODEL}
        for i in range(3)
    ] + [{"gene": "AT_train", "family": "F_train", "chrom": "chr2", "split": "train",
          "observed_log10_tpm": 1., "predicted_log10_tpm": 1., "model": MODULE.MODEL}])


def test_panel_d_writes_paired_data_metrics_and_caption(tmp_path):
    source = tmp_path / "predictions.tsv"
    _rows().to_csv(source, sep="\t", index=False)
    result = MODULE.build_panel(source, tmp_path / "out")
    assert result["status"] == "PASS_PARTIAL_PANEL_D"
    assert result["figure2_complete"] is False
    assert result["paired_gene_count"] == 3
    assert result["metrics"]["pearson_r"] == pytest.approx(1.)
    out = tmp_path / "out"
    assert (out / "nemo_figure2_panel_d.svg").stat().st_size > 0
    assert (out / "nemo_figure2_panel_d.png").stat().st_size > 0
    assert pd.read_csv(out / "panel_d_plotting_data.tsv", sep="\t").gene.tolist() == ["AT0", "AT1", "AT2"]


@pytest.mark.parametrize("mutation,error", [
    (lambda d: pd.concat([d, d.iloc[[0]]], ignore_index=True), "duplicate_or_missing_test_gene"),
    (lambda d: d.assign(predicted_log10_tpm=[float("nan"), 1.25, 2.25, 1.]), "nonfinite_expression_pair"),
])
def test_panel_d_rejects_invalid_paired_inputs(tmp_path, mutation, error):
    source = tmp_path / "predictions.tsv"
    mutation(_rows()).to_csv(source, sep="\t", index=False)
    with pytest.raises(ValueError, match=error):
        MODULE.build_panel(source, tmp_path / "out")
