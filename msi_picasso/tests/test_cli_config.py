"""The CLI passes every parser option through the config cascade."""

import json
from pathlib import Path

from msi_picasso.cli import _CLI_ONLY, _cli_config_source, build_parser
from msi_picasso.config_parser import parse_configurations

_DEFAULTS = json.loads(
    (Path(__file__).parents[1] / "package_data" / "config_default.json").read_text()
)["MSI-PICASSO"]


def test_every_cli_option_is_a_config_key():
    """An option missing from the defaults would be dropped or rejected by the schema."""
    dests = {a.dest for a in build_parser()._actions} - _CLI_ONLY
    assert dests <= set(_DEFAULTS), sorted(dests - set(_DEFAULTS))


def _config(argv, config_file=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    sources = [config_file] if config_file else []
    return parse_configurations(sources + [_cli_config_source(parser, args)])["MSI-PICASSO"]


def test_save_ion_images_flag_reaches_the_config():
    """--save-ion-images used to be silently ignored (it was missing from _TOP_LEVEL_ATTRS)."""
    assert _config(["--maldi-d", "x.d"])["save_ion_images"] is False
    assert _config(["--maldi-d", "x.d", "--save-ion-images"])["save_ion_images"] is True


def test_absent_store_true_flag_does_not_override_config_file(tmp_path):
    cfg = tmp_path / "c.toml"
    cfg.write_text('[MSI-PICASSO]\nsave-ion-images = true\nverbose = true\n')
    out = _config(["--maldi-d", "x.d"], config_file=str(cfg))
    assert out["save_ion_images"] is True
    assert out["verbose"] is True


def test_explicit_cli_value_overrides_config_file(tmp_path):
    cfg = tmp_path / "c.toml"
    cfg.write_text('[MSI-PICASSO]\nmodel = "svm"\n')
    assert _config(["--model", "rbf_svm"], config_file=str(cfg))["model"] == "rbf_svm"
