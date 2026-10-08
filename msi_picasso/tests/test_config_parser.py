"""Tests for parse_configurations() cascade logic."""

import json
from argparse import Namespace
from pathlib import Path

import pytest

from msi_picasso.config_parser import parse_configurations


def test_defaults_returned_when_no_config():
    config = parse_configurations()["MSI-PICASSO"]
    assert config["model"] == "lda"
    assert config["decoy_method"] == "substitution"
    assert config["features_exclude"] == []
    assert config["im2deep_calibration"] == "finetune"
    assert config["r1_seed_percentile"] == pytest.approx(0.10)


def test_toml_overrides_defaults(tmp_path):
    toml = tmp_path / "cfg.toml"
    toml.write_text('[MSI-PICASSO]\nmodel = "svm"\nfeatures_exclude = ["peptide_length"]\n')
    config = parse_configurations([str(toml)])["MSI-PICASSO"]
    assert config["model"] == "svm"
    assert config["features_exclude"] == ["peptide_length"]
    assert config["decoy_method"] == "substitution"


def test_json_overrides_defaults(tmp_path):
    j = tmp_path / "cfg.json"
    j.write_text(json.dumps({"MSI-PICASSO": {"train_fdr": 0.05}}))
    config = parse_configurations([str(j)])["MSI-PICASSO"]
    assert config["train_fdr"] == pytest.approx(0.05)
    assert config["model"] == "lda"


def test_cli_namespace_overrides_file(tmp_path):
    toml = tmp_path / "cfg.toml"
    toml.write_text('[MSI-PICASSO]\nmodel = "svm"\n')
    ns = Namespace(model="lda", train_fdr=None)
    config = parse_configurations([str(toml), ns])["MSI-PICASSO"]
    assert config["model"] == "lda"
    assert config["train_fdr"] == pytest.approx(0.05)


def test_none_does_not_override(tmp_path):
    toml = tmp_path / "cfg.toml"
    toml.write_text('[MSI-PICASSO]\nr1_seed_percentile = 0.05\n')
    ns = Namespace(r1_seed_percentile=None)
    config = parse_configurations([str(toml), ns])["MSI-PICASSO"]
    assert config["r1_seed_percentile"] == pytest.approx(0.05)


def test_schema_rejects_invalid_model(tmp_path):
    j = tmp_path / "bad.json"
    j.write_text(json.dumps({"MSI-PICASSO": {"model": "xgboost"}}))
    with pytest.raises(Exception):
        parse_configurations([str(j)])


def test_features_exclude_unknown_names_allowed():
    ns = Namespace(features_exclude=["nonexistent_feature"])
    config = parse_configurations([ns])["MSI-PICASSO"]
    assert "nonexistent_feature" in config["features_exclude"]


def test_explicit_falsy_zero_is_honored():
    """A legitimate explicit 0 (param with minimum 0) must survive the cascade,
    not be silently dropped to the default (cascade_config falsy-drop bug)."""
    # winner_percentile defaults to 0.02; an explicit 0 must round-trip.
    config = parse_configurations([Namespace(winner_percentile=0)])["MSI-PICASSO"]
    assert config["winner_percentile"] == 0


def test_explicit_matching_ppm_zero_honored():
    """matching_ppm=0 means exact matching / no collision tolerance — a valid choice
    (esp. in raw-query). It must be honored, not silently masked by the default 20.0."""
    config = parse_configurations([Namespace(matching_ppm=0.0)])["MSI-PICASSO"]
    assert config["matching_ppm"] == 0.0
    # a valid positive value is also honored
    config = parse_configurations([Namespace(matching_ppm=8.0)])["MSI-PICASSO"]
    assert config["matching_ppm"] == 8.0


def test_negative_matching_ppm_rejected():
    """A negative matching_ppm is invalid (schema minimum 0) and must raise."""
    with pytest.raises(Exception):
        parse_configurations([Namespace(matching_ppm=-1.0)])


def test_maldi_extraction_section_preserved(tmp_path):
    toml = tmp_path / "cfg.toml"
    toml.write_text('[MSI-PICASSO.maldi_extraction]\nextraction_ppm = 15.0\n')
    config = parse_configurations([str(toml)])["MSI-PICASSO"]
    assert config["maldi_extraction"]["extraction_ppm"] == pytest.approx(15.0)


def test_removed_feature_detection_keys_are_rejected(tmp_path):
    """Feature finding moved to the TIMSImaging fork, so its knobs are gone.

    The schema rejects unknown keys, so a config still carrying one fails loudly
    instead of silently ignoring a setting the user believes is in effect. This
    is the check that catches a stale config from before the move.
    """
    for dead_key in ("ppm_bin = 5.0", "deisotope = true", "peak_prominence = 0.01"):
        toml = tmp_path / f"cfg_{dead_key.split()[0]}.toml"
        toml.write_text(f"[MSI-PICASSO.maldi_extraction]\n{dead_key}\n")
        with pytest.raises(Exception):
            parse_configurations([str(toml)])


def test_im2deep_section_preserved(tmp_path):
    toml = tmp_path / "cfg.toml"
    toml.write_text('[MSI-PICASSO.im2deep]\nfinetune_epochs = 20\n')
    config = parse_configurations([str(toml)])["MSI-PICASSO"]
    assert config["im2deep"]["finetune_epochs"] == 20
    assert config["im2deep"]["finetune_batch_size"] == 64


def test_dict_source_overrides(tmp_path):
    override = {"MSI-PICASSO": {"model_repeats": 3}}
    config = parse_configurations([override])["MSI-PICASSO"]
    assert config["model_repeats"] == 3
    assert config["model"] == "lda"


def test_feature_mzs_keep_round_trip(tmp_path):
    """A path-valued key set in TOML with hyphens reaches the config."""
    toml = tmp_path / "cfg.toml"
    toml.write_text('[MSI-PICASSO]\nfeature-mzs-keep = "/tmp/keep.csv"\n')
    config = parse_configurations([str(toml)])["MSI-PICASSO"]
    assert config["feature_mzs_keep"] == "/tmp/keep.csv"


def test_removed_keys_are_rejected(tmp_path):
    """Options deleted in the cleanup are unknown keys now, so a stale config fails loudly."""
    for dead_key in ("single_round = true", 'model = "qda"', "decoy_split = true",
                     "images_path = \"/tmp/x\"", "lcms_prior_weight = 0.0"):
        toml = tmp_path / "cfg.toml"
        toml.write_text(f"[MSI-PICASSO]\n{dead_key}\n")
        with pytest.raises(Exception):
            parse_configurations([str(toml)])


def test_model_repeats_round_trip(tmp_path):
    """H-fdr-6/F-030: the number of CV partitions averaged is a config knob."""
    toml = tmp_path / "cfg.toml"
    toml.write_text("[MSI-PICASSO]\nmodel-repeats = 10\n")
    config = parse_configurations([str(toml)])["MSI-PICASSO"]
    assert config["model_repeats"] == 10


def test_model_repeats_defaults_to_one():
    """Absent the key, one fixed partition — every result predating this reproduces."""
    assert parse_configurations([{}])["MSI-PICASSO"]["model_repeats"] == 1


def test_substitution_mass_shift_max_da_round_trip(tmp_path):
    """H-fdr-10/F-036: the cap on the substitution mass shift is a config knob."""
    toml = tmp_path / "cfg.toml"
    toml.write_text("[MSI-PICASSO]\nsubstitution_mass_shift_max_da = 120.0\n")
    config = parse_configurations([str(toml)])["MSI-PICASSO"]
    assert config["substitution_mass_shift_max_da"] == 120.0


def test_substitution_mass_shift_max_da_defaults_to_unset():
    """Absent the key there is no cap, so earlier results reproduce."""
    assert parse_configurations([{}])["MSI-PICASSO"]["substitution_mass_shift_max_da"] is None
