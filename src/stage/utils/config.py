"""Configuration loading and path resolution utilities."""

import os
from copy import deepcopy
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PATH_PROFILE_ENV = "STAGE_PATH_PROFILE"


def load_yaml(path):
    """Load a YAML file from an absolute path or project-relative path."""
    path = Path(path)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def resolve_path(path_str):
    """Resolve an absolute or project-relative path."""
    path = Path(path_str)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def load_paths():
    """Load configured project paths."""
    return load_yaml("configs/paths.yaml")


def _deep_merge(base, override):
    """Recursively merge override into base without modifying either input."""
    result = deepcopy(base)

    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)

    return result


def get_active_profile():
    """Return the active path profile name, or None for the default paths."""
    profile = os.environ.get(PATH_PROFILE_ENV)

    if profile is None or profile.strip() == "":
        return None

    return profile.strip()


def get_active_paths():
    """Return path configuration after applying the active path profile."""
    profile = get_active_profile()

    if profile is None:
        return {
            key: value
            for key, value in PATHS.items()
            if key != "path_profiles"
        }

    profiles = PATHS.get("path_profiles", {})

    if profile not in profiles:
        available = ", ".join(sorted(profiles.keys())) or "<none>"
        raise KeyError(
            f"Unknown {PATH_PROFILE_ENV}={profile!r}. "
            f"Available profiles: {available}"
        )

    base = {
        key: value
        for key, value in PATHS.items()
        if key != "path_profiles"
    }
    return _deep_merge(base, profiles[profile])


PATHS = load_paths()


def get_path(*keys):
    """Return a configured path by nested keys.

    If STAGE_PATH_PROFILE is set, matching values from configs/paths.yaml
    path_profiles are overlaid on the default path configuration.
    """
    value = get_active_paths()

    for key in keys:
        value = value[key]

    return resolve_path(value)
