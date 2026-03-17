"""
Visualization utilities using matplotlib.
All functions return matplotlib Figure objects for flexible display/saving.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.figure import Figure

# Regime color palette
REGIME_COLORS: Dict[str, str] = {
    "trend_up":       "#2ecc71",   # green
    "trend_down":     "#e74c3c",   # red
    "range_bound":    "#3498db",   # blue
    "mean_reverting": "#9b59b6",   # purple
    "high_volatility":"#e67e22",   # orange
    "low_volatility": "#95a5a6",   # grey
    "event_risk":     "#1a1a2e",   # dark
}


def _style() -> None:
    plt.style.use("seaborn-v0_8-darkgrid")
    plt.rcParams.update({"figure.dpi": 110, "font.size": 10})


def plot_equity_curve(
    equity: pd.Series,
    title: str = "Equity Curve",
    benchmark: Optional[pd.Series] = None,
) -> Figure:
    _style()
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})

    ax1, ax2 = axes
    ax1.plot(equity.index, equity.values, linewidth=1.5, label="Strategy", color="#2ecc71")
    if benchmark is not None:
        bench_scaled = benchmark / benchmark.iloc[0] * equity.iloc[0]
        ax1.plot(bench_scaled.index, bench_scaled.values, linewidth=1,
                 label="Benchmark", color="#95a5a6", linestyle="--")
    ax1.set_title(title, fontsize=13)
    ax1.set_ylabel("Equity")
    ax1.legend()

    # Drawdown panel
    roll_max = equity.cummax()
    drawdown = (equity - roll_max) / roll_max * 100
    ax2.fill_between(drawdown.index, drawdown.values, 0, color="#e74c3c", alpha=0.4)
    ax2.set_ylabel("Drawdown (%)")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def plot_regime_timeline(
    price: pd.Series,
    regimes: pd.Series,
    title: str = "Regime Timeline",
) -> Figure:
    """
    Overlay regime coloured bands on price chart.

    Args:
        price: close price series with DatetimeIndex
        regimes: series of regime label strings aligned to price.index
    """
    _style()
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(price.index, price.values, color="black", linewidth=0.8, zorder=3)

    # Shade regime bands
    prev_regime = None
    prev_start = price.index[0]
    for ts, regime in regimes.items():
        if regime != prev_regime:
            if prev_regime is not None:
                color = REGIME_COLORS.get(prev_regime, "#cccccc")
                ax.axvspan(prev_start, ts, alpha=0.25, color=color, zorder=1)
            prev_regime = regime
            prev_start = ts
    if prev_regime is not None:
        color = REGIME_COLORS.get(prev_regime, "#cccccc")
        ax.axvspan(prev_start, price.index[-1], alpha=0.25, color=color, zorder=1)

    # Legend
    patches = [mpatches.Patch(color=c, label=r, alpha=0.5)
               for r, c in REGIME_COLORS.items()
               if r in regimes.unique()]
    ax.legend(handles=patches, loc="upper left", fontsize=8, ncol=4)
    ax.set_title(title)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def plot_regime_probabilities(
    prob_df: pd.DataFrame,
    title: str = "Regime Probabilities",
) -> Figure:
    """
    Stacked area chart of regime probabilities over time.

    Args:
        prob_df: DataFrame with DatetimeIndex, columns = regime labels, values in [0,1]
    """
    _style()
    fig, ax = plt.subplots(figsize=(16, 4))
    colors = [REGIME_COLORS.get(c, "#aaaaaa") for c in prob_df.columns]
    ax.stackplot(prob_df.index, prob_df.values.T, labels=prob_df.columns,
                 colors=colors, alpha=0.7)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8, ncol=4)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def plot_trade_distribution(
    trades: pd.DataFrame,
    title: str = "Trade P&L Distribution",
) -> Figure:
    """
    Histogram + KDE of trade returns.

    Args:
        trades: DataFrame with 'pnl_pct' column.
    """
    _style()
    from scipy.stats import gaussian_kde  # optional; catch import error gracefully
    fig, ax = plt.subplots(figsize=(10, 5))
    pnl = trades["pnl_pct"].dropna()
    ax.hist(pnl, bins=40, density=True, alpha=0.5, color="#3498db", label="Trades")
    try:
        kde = gaussian_kde(pnl)
        xs = np.linspace(pnl.min(), pnl.max(), 300)
        ax.plot(xs, kde(xs), color="#2c3e50", linewidth=2, label="KDE")
    except Exception:
        pass
    ax.axvline(0, color="red", linestyle="--", linewidth=1)
    ax.axvline(pnl.mean(), color="green", linestyle="--", linewidth=1,
               label=f"Mean {pnl.mean():.2f}%")
    ax.set_xlabel("P&L (%)")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_performance_by_regime(
    metrics: pd.DataFrame,
    metric: str = "profit_factor",
    title: Optional[str] = None,
) -> Figure:
    """
    Bar chart of a chosen metric broken down by regime.

    Args:
        metrics: DataFrame indexed by regime label with a column `metric`.
        metric: column to plot.
    """
    _style()
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [REGIME_COLORS.get(r, "#aaaaaa") for r in metrics.index]
    ax.bar(metrics.index, metrics[metric], color=colors, alpha=0.8, edgecolor="white")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(title or f"{metric.replace('_',' ').title()} by Regime")
    ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.8)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    return fig


def plot_monthly_returns(
    equity: pd.Series,
    title: str = "Monthly Returns Heatmap",
) -> Figure:
    """Seaborn-style monthly returns heatmap."""
    _style()
    monthly = equity.resample("ME").last().pct_change().dropna()
    df = monthly.to_frame("ret")
    df["year"]  = df.index.year
    df["month"] = df.index.month
    pivot = df.pivot(index="year", columns="month", values="ret") * 100
    pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun",
                     "Jul","Aug","Sep","Oct","Nov","Dec"][: len(pivot.columns)]

    fig, ax = plt.subplots(figsize=(14, max(3, len(pivot) * 0.6 + 1)))
    vmax = max(abs(pivot.min().min()), abs(pivot.max().max()))
    im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.iloc[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                        fontsize=8, color="black")
    plt.colorbar(im, ax=ax, label="Return (%)")
    ax.set_title(title)
    fig.tight_layout()
    return fig
