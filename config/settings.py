"""
Global system settings loaded from environment variables.
All user-facing configuration lives here.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass
class DataSettings:
    provider: str = "yfinance"            # yfinance | alpha_vantage | twelve_data | polygon | csv
    cache_dir: Path = BASE_DIR / "data_cache"
    market_csv_dir: Path = BASE_DIR / "data" / "market"
    news_csv_path: Path = BASE_DIR / "data" / "news" / "events.csv"
    default_timeframe: str = "4h"        # pandas-compatible offset alias
    default_lookback_days: int = 730     # 2 years of history by default


@dataclass
class RegimeSettings:
    n_hmm_states: int = 4
    hmm_covariance_type: str = "full"    # full | diag | tied | spherical
    hmm_n_iter: int = 200
    regime_lookback_bars: int = 252      # bars used to fit HMM
    min_regime_confidence: float = 0.50  # below this → "uncertain"
    use_ensemble: bool = True


@dataclass
class NewsFilterSettings:
    pre_event_minutes: int = 60
    post_event_minutes: int = 30
    block_high_impact: bool = True
    block_medium_impact: bool = False
    currencies_of_interest: List[str] = field(
        default_factory=lambda: ["USD", "EUR", "GBP", "JPY", "AUD", "CHF", "CAD"]
    )


@dataclass
class SessionSettings:
    timezone: str = "UTC"
    sessions: dict = field(default_factory=lambda: {
        "sydney":  {"start": "21:00", "end": "06:00"},
        "tokyo":   {"start": "00:00", "end": "09:00"},
        "london":  {"start": "07:00", "end": "16:00"},
        "new_york": {"start": "12:00", "end": "21:00"},
    })
    preferred_sessions: List[str] = field(
        default_factory=lambda: ["london", "new_york"]
    )


@dataclass
class RiskSettings:
    default_risk_pct: float = 2.0        # % of equity per trade
    max_open_trades: int = 10            # global across all instruments (10 reduces crypto overexposure)
    max_daily_loss_pct: float = 5.0
    max_drawdown_pct: float = 20.0
    volatility_target_pct: Optional[float] = None   # None = disabled
    max_asset_exposure_pct: float = 10.0  # % of portfolio per asset
    min_rr_ratio: float = 1.5            # minimum reward:risk ratio to accept a trade


@dataclass
class OptimizationSettings:
    method: str = "optuna"               # grid | random | optuna
    n_trials: int = 200                  # optuna trials or random combos
    n_jobs: int = -1                     # -1 = all cores
    cv_folds: int = 5
    purge_gap_bars: int = 10             # embargo bars between train/test
    primary_metric: str = "calmar"       # sharpe | sortino | calmar | profit_factor
    min_trades: int = 10                 # lower for 4H (fewer bars than 1H)


@dataclass
class ExecutionSettings:
    signal_output_path: Path = BASE_DIR / "signals" / "latest_signal.json"
    mt5_files_path: Optional[Path] = None       # MQL5\Files\ mirror (auto-detected)
    paper_mode: bool = True
    slippage_pct: float = 0.0002         # 0.02% per side
    commission_pct: float = 0.0001       # 0.01% per side


def _detect_mt5_files_path() -> Optional[Path]:
    """
    Auto-detect MT5 MQL5/Files folder from MetaQuotes appdata.

    Prefers the terminal that has Experts/ with .mq5 files (active installation).
    Falls back to most recently modified terminal.
    """
    mq_root = Path(os.path.expandvars(r"%APPDATA%")) / "MetaQuotes" / "Terminal"
    if not mq_root.exists():
        return None

    candidates = []
    for terminal_dir in mq_root.iterdir():
        if not terminal_dir.is_dir():
            continue
        files_dir = terminal_dir / "MQL5" / "Files"
        experts_dir = terminal_dir / "MQL5" / "Experts"
        if not files_dir.is_dir():
            continue
        has_experts = experts_dir.is_dir() and any(experts_dir.glob("*.mq5"))
        try:
            mtime = terminal_dir.stat().st_mtime
        except OSError:
            mtime = 0
        candidates.append((has_experts, mtime, files_dir))

    if not candidates:
        return None

    # Sort: has_experts=True first, then by most recent modification
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return candidates[0][2]


@dataclass
class Settings:
    data: DataSettings = field(default_factory=DataSettings)
    regime: RegimeSettings = field(default_factory=RegimeSettings)
    news_filter: NewsFilterSettings = field(default_factory=NewsFilterSettings)
    session: SessionSettings = field(default_factory=SessionSettings)
    risk: RiskSettings = field(default_factory=RiskSettings)
    optimization: OptimizationSettings = field(default_factory=OptimizationSettings)
    execution: ExecutionSettings = field(default_factory=ExecutionSettings)

    @classmethod
    def from_env(cls) -> "Settings":
        """Override defaults from environment variables."""
        s = cls()
        # Data
        if v := os.getenv("MARKET_DATA_PROVIDER"):
            s.data.provider = v
        if v := os.getenv("DATA_CACHE_DIR"):
            s.data.cache_dir = Path(v)
        if v := os.getenv("MARKET_DATA_CSV_DIR"):
            s.data.market_csv_dir = Path(v)
        if v := os.getenv("NEWS_DATA_CSV_PATH"):
            s.data.news_csv_path = Path(v)
        # Session
        if v := os.getenv("TIMEZONE"):
            s.session.timezone = v
        # Execution
        if v := os.getenv("MT5_SIGNAL_OUTPUT_PATH"):
            s.execution.signal_output_path = Path(v)
        if v := os.getenv("MT5_FILES_PATH"):
            s.execution.mt5_files_path = Path(v)
        else:
            # Auto-detect MT5 data folder
            s.execution.mt5_files_path = _detect_mt5_files_path()
        # Risk
        if v := os.getenv("DEFAULT_RISK_PCT"):
            s.risk.default_risk_pct = float(v)
        return s


# Module-level singleton loaded from env
SETTINGS = Settings.from_env()
