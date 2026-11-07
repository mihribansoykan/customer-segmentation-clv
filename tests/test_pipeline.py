"""End-to-end smoke test of the pipeline on synthetic transactions (no download)."""

import json

from retail_clv import data, pipeline


def test_pipeline_end_to_end(tx, tmp_path, monkeypatch):
    raw = tx.drop(columns=["is_return", "revenue"])
    monkeypatch.setattr(data, "download", lambda force=False: tmp_path / "unused.parquet")
    monkeypatch.setattr(data, "load_raw", lambda path=None: raw.copy())
    monkeypatch.setattr(pipeline, "REPORTS", tmp_path)
    monkeypatch.setattr(pipeline, "FIGS", tmp_path / "figures")

    metrics = pipeline.run()

    saved = json.loads((tmp_path / "metrics.json").read_text())
    assert saved["data"]["customers"] == raw["customer_id"].nunique()
    assert set(saved["clv"]["results"]) == {
        "Naive (last 6 months)",
        "BG/NBD + Gamma-Gamma",
        "LightGBM (behavioural)",
        "LightGBM + BTYD features",
    }
    for res in metrics["clv"]["results"].values():
        assert 0 <= res["top20_capture"] <= 1
    assert (tmp_path / "segment_summary.csv").exists()
    assert len(list((tmp_path / "figures").glob("*.png"))) == 8
