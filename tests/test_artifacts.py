"""Tests for test artifacts."""

from __future__ import annotations

from rfm_pipeline.artifacts import PipelineManifest, build_position_map, make_metadata_frame


def test_build_position_map_preserves_original_order():
    assert build_position_map(["b", "a"]) == {"b": 0, "a": 1}


def test_make_metadata_frame_records_positions():
    df = make_metadata_frame(["x1", "x2"], "feature_name")
    assert df.to_dict(orient="records") == [
        {"feature_name": "x1", "original_position": 0},
        {"feature_name": "x2", "original_position": 1},
    ]


def test_pipeline_manifest_adds_position_maps():
    manifest = PipelineManifest(
        dataset_tag="demo",
        n_all_input_features=3,
        n_selected_features=2,
        n_retained_features=2,
        n_outputs=4,
        all_input_features=["x1", "x2", "x3"],
        selected_features=["x2", "x3"],
        retained_features=["x2", "x3"],
        output_names=["y1", "y2", "y3", "y4"],
        files={"manifest": "model_manifest.json"},
    )
    payload = manifest.to_dict()
    assert payload["selected_input_position_map"] == {"x2": 0, "x3": 1}
    assert payload["output_position_map"]["y4"] == 3
