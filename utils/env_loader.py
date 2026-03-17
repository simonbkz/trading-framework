"""
Environment variable loader with validation.

Usage:
    from utils.env_loader import load_env, require_env, optional_env

Call load_env() once at startup to load .env file if present.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_env(dotenv_path: Optional[Path] = None) -> None:
    """
    Load .env file into os.environ.
    Uses python-dotenv if available; gracefully skips if not installed.
    Does NOT overwrite existing environment variables.
    """
    if dotenv_path is None:
        dotenv_path = Path(__file__).resolve().parent.parent / ".env"

    if not dotenv_path.exists():
        return

    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=str(dotenv_path), override=False)
    except ImportError:
        # Parse manually — handles key=value and export key=value
        with open(dotenv_path) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                line = line.removeprefix("export").strip()
                if "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value


def require_env(key: str, description: str = "") -> str:
    """
    Return the value of an environment variable.
    Raises EnvironmentError with a clear message if missing or empty.
    """
    value = os.getenv(key, "").strip()
    if not value:
        hint = f" ({description})" if description else ""
        raise EnvironmentError(
            f"Required environment variable '{key}' is not set{hint}.\n"
            f"Add it to your .env file or export it before running."
        )
    return value


def optional_env(key: str, default: Any = None) -> Any:
    """Return the value of an optional environment variable or the default."""
    return os.getenv(key, default)


def validate_required_vars(required: List[Dict[str, str]]) -> None:
    """
    Validate a list of required environment variables.

    Args:
        required: list of {"key": "VAR_NAME", "description": "..."}

    Raises:
        EnvironmentError listing all missing variables at once.
    """
    missing = []
    for entry in required:
        key = entry["key"]
        if not os.getenv(key, "").strip():
            desc = entry.get("description", "")
            missing.append(f"  {key}" + (f": {desc}" if desc else ""))

    if missing:
        raise EnvironmentError(
            "The following required environment variables are missing:\n"
            + "\n".join(missing)
            + "\n\nSee .env.example for the full list of variables."
        )


# Required variables per provider — checked lazily when the provider is used
PROVIDER_REQUIRED_VARS: Dict[str, List[Dict[str, str]]] = {
    "alpha_vantage": [
        {"key": "ALPHA_VANTAGE_API_KEY", "description": "Alpha Vantage API key"},
    ],
    "twelve_data": [
        {"key": "TWELVE_DATA_API_KEY", "description": "Twelve Data API key"},
    ],
    "polygon": [
        {"key": "POLYGON_API_KEY", "description": "Polygon.io API key"},
    ],
    "trading_economics": [
        {"key": "TRADING_ECONOMICS_API_KEY",    "description": "Trading Economics API key"},
        {"key": "TRADING_ECONOMICS_API_SECRET", "description": "Trading Economics API secret"},
    ],
}


def validate_provider_env(provider_name: str) -> None:
    """Raise if the required env vars for a provider are not set."""
    required = PROVIDER_REQUIRED_VARS.get(provider_name, [])
    validate_required_vars(required)
