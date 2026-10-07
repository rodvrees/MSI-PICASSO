"""Cascade configuration for MSI-PICASSO.

Priority (lowest to highest):
  1. package_data/config_default.json  — all defaults
  2. User --config-file (JSON or TOML)
  3. CLI arguments (explicit only; None values never override lower-priority sources)
"""

import json
import tomllib
from argparse import Namespace
from pathlib import Path

import jsonschema

_SCHEMA = Path(__file__).parent / "package_data" / "config_schema.json"
_DEFAULT = Path(__file__).parent / "package_data" / "config_default.json"


def _normalize_keys(d: dict) -> dict:
    """Recursively replace hyphens with underscores in dict keys.

    Allows TOML/JSON config files to use either ``maldi-raw`` or ``maldi_raw``
    — both map to the same schema key.
    """
    return {
        k.replace("-", "_"): (_normalize_keys(v) if isinstance(v, dict) else v)
        for k, v in d.items()
    }


def _normalize_config(d: dict) -> dict:
    """Normalize a top-level config dict.

    The top-level section key is the literal ``"MSI-PICASSO"`` (with a hyphen),
    so it must NOT be hyphen-normalized — only the keys *inside* each section
    are normalized (e.g. ``maldi-raw`` → ``maldi_raw``).
    """
    return {
        k: (_normalize_keys(v) if isinstance(v, dict) else v)
        for k, v in d.items()
    }


def _merge(base: dict, over: dict) -> None:
    """Deep-merge ``over`` into ``base`` in place.

    Any non-None value wins, including falsy ones (an explicit ``0`` is honored).
    ``None`` means "unset" and only fills a key that does not exist yet.
    """
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge(base[k], v)
        elif v is not None or k not in base:
            base[k] = v


def parse_configurations(configurations=None) -> dict:
    """Merge config sources and return the resolved config dict.

    Parameters
    ----------
    configurations
        Ordered list of config sources to layer on top of defaults.
        Each item may be:
        - a file path (str or Path) to a JSON or TOML config file
        - an argparse.Namespace (CLI args)
        - a plain dict

    Returns
    -------
    dict
        Merged config with top-level key ``"MSI-PICASSO"``.
    """
    result = json.loads(_DEFAULT.read_text())
    for config in (configurations or []):
        if isinstance(config, dict):
            nc = _normalize_config(config)
        elif isinstance(config, (str, Path)):
            p = Path(config)
            if p.suffix.lower() == ".json":
                nc = _normalize_config(json.loads(p.read_text()))
            elif p.suffix.lower() in (".toml", ".tml"):
                nc = _normalize_config(tomllib.loads(p.read_text()))
            else:
                raise ValueError(
                    f"Unsupported config file format: {p.suffix!r}. "
                    "Use .json or .toml."
                )
        elif isinstance(config, Namespace):
            nc = {"MSI-PICASSO": dict(vars(config))}
        else:
            raise TypeError(
                f"Unsupported config source type: {type(config).__name__}. "
                "Expected a file path, argparse.Namespace, or dict."
            )
        _merge(result, nc)

    # An out-of-range explicit value (e.g. matching_ppm=0, forbidden by the schema's
    # exclusiveMinimum) raises here instead of being silently masked by the default.
    jsonschema.validate(result, json.loads(_SCHEMA.read_text()))
    return result
