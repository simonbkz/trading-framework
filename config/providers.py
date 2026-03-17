"""
Data provider registry and per-provider capability declarations.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ProviderConfig:
    name: str
    env_key_var: Optional[str]          # env var holding the API key (None = no key needed)
    env_secret_var: Optional[str]       # optional secret
    supports_forex: bool = True
    supports_crypto: bool = True
    supports_indices: bool = True
    supports_commodities: bool = True
    max_bars_per_request: int = 5000
    rate_limit_per_minute: int = 60
    free_tier: bool = True
    notes: str = ""

    @property
    def api_key(self) -> Optional[str]:
        if self.env_key_var:
            return os.getenv(self.env_key_var)
        return None

    @property
    def api_secret(self) -> Optional[str]:
        if self.env_secret_var:
            return os.getenv(self.env_secret_var)
        return None

    @property
    def is_available(self) -> bool:
        """True if provider can be used (key present or no key needed)."""
        if self.env_key_var is None:
            return True
        return bool(self.api_key)


PROVIDERS: Dict[str, ProviderConfig] = {
    "yfinance": ProviderConfig(
        name="yfinance",
        env_key_var=None,
        env_secret_var=None,
        supports_indices=True,
        max_bars_per_request=10000,
        rate_limit_per_minute=2000,
        free_tier=True,
        notes="No key required. Best free option. Use as default/fallback.",
    ),
    "alpha_vantage": ProviderConfig(
        name="alpha_vantage",
        env_key_var="ALPHA_VANTAGE_API_KEY",
        env_secret_var=None,
        max_bars_per_request=1000,
        rate_limit_per_minute=5,   # free tier
        free_tier=True,
        notes="Free tier: 5 req/min, 500 req/day. Premium increases limits.",
    ),
    "twelve_data": ProviderConfig(
        name="twelve_data",
        env_key_var="TWELVE_DATA_API_KEY",
        env_secret_var=None,
        max_bars_per_request=5000,
        rate_limit_per_minute=8,   # free tier
        free_tier=True,
        notes="Free tier: 800 req/day. Good forex and crypto coverage.",
    ),
    "polygon": ProviderConfig(
        name="polygon",
        env_key_var="POLYGON_API_KEY",
        env_secret_var=None,
        supports_forex=True,
        supports_crypto=True,
        supports_indices=False,
        max_bars_per_request=50000,
        rate_limit_per_minute=5,
        free_tier=True,
        notes="Free tier: equities/forex/crypto. No index futures.",
    ),
    "csv": ProviderConfig(
        name="csv",
        env_key_var=None,
        env_secret_var=None,
        max_bars_per_request=999999,
        rate_limit_per_minute=999999,
        free_tier=True,
        notes="CSV fallback. Files must be in MARKET_DATA_CSV_DIR.",
    ),
    "trading_economics": ProviderConfig(
        name="trading_economics",
        env_key_var="TRADING_ECONOMICS_API_KEY",
        env_secret_var="TRADING_ECONOMICS_API_SECRET",
        supports_forex=False,
        supports_crypto=False,
        supports_indices=False,
        free_tier=False,
        notes="Used for economic calendar / news events only.",
    ),
}


def get_provider(name: str) -> ProviderConfig:
    if name not in PROVIDERS:
        raise ValueError(f"Unknown provider '{name}'. Available: {list(PROVIDERS)}")
    return PROVIDERS[name]


def available_providers() -> List[str]:
    return [name for name, p in PROVIDERS.items() if p.is_available]
