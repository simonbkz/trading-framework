"""
Asset universe definitions and per-asset metadata.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class AssetConfig:
    symbol: str
    display_name: str
    asset_class: str          # crypto | commodity | index | forex
    base_currency: str
    quote_currency: str
    pip_size: float           # smallest price move tracked
    pip_value_usd: float      # USD value of 1 pip per 1 standard lot (or unit)
    typical_spread_pips: float
    yfinance_ticker: str
    twelve_data_symbol: str
    alpha_vantage_symbol: str
    polygon_ticker: str
    sessions: List[str]       # preferred trading sessions
    min_atr_filter: float = 0.0   # ATR below this → skip (too quiet)
    leverage_max: int = 100


ASSET_UNIVERSE: Dict[str, AssetConfig] = {
    "BTCUSD": AssetConfig(
        symbol="BTCUSD",
        display_name="Bitcoin / USD",
        asset_class="crypto",
        base_currency="BTC",
        quote_currency="USD",
        pip_size=1.0,
        pip_value_usd=1.0,
        typical_spread_pips=5.0,
        yfinance_ticker="BTC-USD",
        twelve_data_symbol="BTC/USD",
        alpha_vantage_symbol="BTCUSD",
        polygon_ticker="X:BTCUSD",
        sessions=["london", "new_york", "sydney", "tokyo"],
        min_atr_filter=100.0,
        leverage_max=10,
    ),
    "XAUUSD": AssetConfig(
        symbol="XAUUSD",
        display_name="Gold / USD",
        asset_class="commodity",
        base_currency="XAU",
        quote_currency="USD",
        pip_size=0.01,
        pip_value_usd=1.0,
        typical_spread_pips=3.0,
        yfinance_ticker="GC=F",
        twelve_data_symbol="XAU/USD",
        alpha_vantage_symbol="XAUUSD",
        polygon_ticker="C:XAUUSD",
        sessions=["london", "new_york"],
        min_atr_filter=5.0,
        leverage_max=100,
    ),
    "Nasdaq": AssetConfig(
        symbol="Nasdaq",
        display_name="Nasdaq 100",
        asset_class="index",
        base_currency="USD",
        quote_currency="USD",
        pip_size=0.1,
        pip_value_usd=1.0,
        typical_spread_pips=2.0,
        yfinance_ticker="NQ=F",
        twelve_data_symbol="NDX",
        alpha_vantage_symbol="NDX",
        polygon_ticker="I:NDX",
        sessions=["new_york"],
        min_atr_filter=20.0,
        leverage_max=20,
    ),
    "EURUSD": AssetConfig(
        symbol="EURUSD",
        display_name="Euro / USD",
        asset_class="forex",
        base_currency="EUR",
        quote_currency="USD",
        pip_size=0.0001,
        pip_value_usd=10.0,
        typical_spread_pips=1.0,
        yfinance_ticker="EURUSD=X",
        twelve_data_symbol="EUR/USD",
        alpha_vantage_symbol="EURUSD",
        polygon_ticker="C:EURUSD",
        sessions=["london", "new_york"],
        min_atr_filter=0.0005,
        leverage_max=100,
    ),
    "GBPUSD": AssetConfig(
        symbol="GBPUSD",
        display_name="Pound / USD",
        asset_class="forex",
        base_currency="GBP",
        quote_currency="USD",
        pip_size=0.0001,
        pip_value_usd=10.0,
        typical_spread_pips=1.5,
        yfinance_ticker="GBPUSD=X",
        twelve_data_symbol="GBP/USD",
        alpha_vantage_symbol="GBPUSD",
        polygon_ticker="C:GBPUSD",
        sessions=["london", "new_york"],
        min_atr_filter=0.0006,
        leverage_max=100,
    ),
    "USDJPY": AssetConfig(
        symbol="USDJPY",
        display_name="USD / Japanese Yen",
        asset_class="forex",
        base_currency="USD",
        quote_currency="JPY",
        pip_size=0.01,
        pip_value_usd=6.5,  # approximate
        typical_spread_pips=1.5,
        yfinance_ticker="USDJPY=X",
        twelve_data_symbol="USD/JPY",
        alpha_vantage_symbol="USDJPY",
        polygon_ticker="C:USDJPY",
        sessions=["tokyo", "london", "new_york"],
        min_atr_filter=0.20,
        leverage_max=100,
    ),
    "AUDUSD": AssetConfig(
        symbol="AUDUSD",
        display_name="Australian Dollar / USD",
        asset_class="forex",
        base_currency="AUD",
        quote_currency="USD",
        pip_size=0.0001,
        pip_value_usd=10.0,
        typical_spread_pips=1.5,
        yfinance_ticker="AUDUSD=X",
        twelve_data_symbol="AUD/USD",
        alpha_vantage_symbol="AUDUSD",
        polygon_ticker="C:AUDUSD",
        sessions=["sydney", "london"],
        min_atr_filter=0.0004,
        leverage_max=100,
    ),
    "ETHUSD": AssetConfig(
        symbol="ETHUSD",
        display_name="Ethereum / USD",
        asset_class="crypto",
        base_currency="ETH",
        quote_currency="USD",
        pip_size=0.01,
        pip_value_usd=1.0,
        typical_spread_pips=2.0,
        yfinance_ticker="ETH-USD",
        twelve_data_symbol="ETH/USD",
        alpha_vantage_symbol="ETHUSD",
        polygon_ticker="X:ETHUSD",
        sessions=["london", "new_york", "sydney", "tokyo"],
        min_atr_filter=10.0,
        leverage_max=10,
    ),
    "USDCAD": AssetConfig(
        symbol="USDCAD",
        display_name="USD / Canadian Dollar",
        asset_class="forex",
        base_currency="USD",
        quote_currency="CAD",
        pip_size=0.0001,
        pip_value_usd=7.5,
        typical_spread_pips=1.5,
        yfinance_ticker="USDCAD=X",
        twelve_data_symbol="USD/CAD",
        alpha_vantage_symbol="USDCAD",
        polygon_ticker="C:USDCAD",
        sessions=["london", "new_york"],
        min_atr_filter=0.0004,
        leverage_max=100,
    ),
    "EURGBP": AssetConfig(
        symbol="EURGBP",
        display_name="Euro / Pound",
        asset_class="forex",
        base_currency="EUR",
        quote_currency="GBP",
        pip_size=0.0001,
        pip_value_usd=12.0,
        typical_spread_pips=1.0,
        yfinance_ticker="EURGBP=X",
        twelve_data_symbol="EUR/GBP",
        alpha_vantage_symbol="EURGBP",
        polygon_ticker="C:EURGBP",
        sessions=["london", "new_york"],
        min_atr_filter=0.0003,
        leverage_max=100,
    ),
    "NZDUSD": AssetConfig(
        symbol="NZDUSD",
        display_name="New Zealand Dollar / USD",
        asset_class="forex",
        base_currency="NZD",
        quote_currency="USD",
        pip_size=0.0001,
        pip_value_usd=10.0,
        typical_spread_pips=1.5,
        yfinance_ticker="NZDUSD=X",
        twelve_data_symbol="NZD/USD",
        alpha_vantage_symbol="NZDUSD",
        polygon_ticker="C:NZDUSD",
        sessions=["sydney", "london"],
        min_atr_filter=0.0004,
        leverage_max=100,
    ),
    "GBPJPY": AssetConfig(
        symbol="GBPJPY",
        display_name="Pound / Japanese Yen",
        asset_class="forex",
        base_currency="GBP",
        quote_currency="JPY",
        pip_size=0.01,
        pip_value_usd=6.5,
        typical_spread_pips=3.0,
        yfinance_ticker="GBPJPY=X",
        twelve_data_symbol="GBP/JPY",
        alpha_vantage_symbol="GBPJPY",
        polygon_ticker="C:GBPJPY",
        sessions=["tokyo", "london", "new_york"],
        min_atr_filter=0.20,
        leverage_max=100,
    ),
    "EURJPY": AssetConfig(
        symbol="EURJPY",
        display_name="Euro / Japanese Yen",
        asset_class="forex",
        base_currency="EUR",
        quote_currency="JPY",
        pip_size=0.01,
        pip_value_usd=6.5,
        typical_spread_pips=2.0,
        yfinance_ticker="EURJPY=X",
        twelve_data_symbol="EUR/JPY",
        alpha_vantage_symbol="EURJPY",
        polygon_ticker="C:EURJPY",
        sessions=["tokyo", "london", "new_york"],
        min_atr_filter=0.20,
        leverage_max=100,
    ),
    "SP500": AssetConfig(
        symbol="SP500",
        display_name="S&P 500",
        asset_class="index",
        base_currency="USD",
        quote_currency="USD",
        pip_size=0.1,
        pip_value_usd=1.0,
        typical_spread_pips=1.5,
        yfinance_ticker="ES=F",
        twelve_data_symbol="SPX",
        alpha_vantage_symbol="SPX",
        polygon_ticker="I:SPX",
        sessions=["new_york"],
        min_atr_filter=10.0,
        leverage_max=20,
    ),
    "XAGUSD": AssetConfig(
        symbol="XAGUSD",
        display_name="Silver / USD",
        asset_class="commodity",
        base_currency="XAG",
        quote_currency="USD",
        pip_size=0.001,
        pip_value_usd=5.0,
        typical_spread_pips=3.0,
        yfinance_ticker="SI=F",
        twelve_data_symbol="XAG/USD",
        alpha_vantage_symbol="XAGUSD",
        polygon_ticker="C:XAGUSD",
        sessions=["london", "new_york"],
        min_atr_filter=0.10,
        leverage_max=50,
    ),
    "USOIL": AssetConfig(
        symbol="USOIL",
        display_name="WTI Crude Oil",
        asset_class="commodity",
        base_currency="OIL",
        quote_currency="USD",
        pip_size=0.01,
        pip_value_usd=10.0,
        typical_spread_pips=3.0,
        yfinance_ticker="CL=F",
        twelve_data_symbol="WTI/USD",
        alpha_vantage_symbol="USOIL",
        polygon_ticker="C:USOIL",
        sessions=["london", "new_york"],
        min_atr_filter=0.20,
        leverage_max=30,
    ),
}

DEFAULT_ASSETS: List[str] = list(ASSET_UNIVERSE.keys())


def get_asset(symbol: str) -> AssetConfig:
    if symbol not in ASSET_UNIVERSE:
        raise ValueError(f"Unknown asset '{symbol}'. Available: {list(ASSET_UNIVERSE)}")
    return ASSET_UNIVERSE[symbol]


def get_yf_ticker(symbol: str) -> str:
    return get_asset(symbol).yfinance_ticker


def assets_by_class(asset_class: str) -> List[str]:
    return [s for s, a in ASSET_UNIVERSE.items() if a.asset_class == asset_class]
