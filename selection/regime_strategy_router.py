"""
RegimeStrategyRouter — the master orchestrator.

Runs the full signal pipeline:
    regime detection → filters → asset selection →
    strategy selection → side selection →
    parameter loading → signal generation → output
"""
from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from config.settings import SETTINGS
from regimes.regime_service import RegimeService
from filters.tradability_filter import TradabilityFilter
from selection.asset_selector import AssetSelector
from selection.strategy_selector import StrategySelector
from selection.side_selector import SideSelector
from strategies.strategy_factory import get_strategy, ParameterRegistry
from optimization.parameter_store import ParameterStore
from strategies.base_strategy import TradeProposal
from utils.logger import get_logger

log = get_logger(__name__)


class RegimeStrategyRouter:
    """
    End-to-end pipeline from OHLCV data to a list of TradeProposals.

    Usage:
        router = RegimeStrategyRouter(regime_svc, param_store, assets)
        proposals = router.route(market_data_dict)
    """

    def __init__(
        self,
        regime_service: RegimeService,
        parameter_store: Optional[ParameterStore] = None,
        asset_universe: Optional[List[str]] = None,
        tradability_filter: Optional[TradabilityFilter] = None,
    ):
        self.regime_svc   = regime_service
        self.param_store  = parameter_store or ParameterStore()
        self.assets       = asset_universe or list(pd.Series(dtype=str))
        self.tf_filter    = tradability_filter or TradabilityFilter()
        self.asset_sel    = AssetSelector(asset_universe=self.assets)
        self.strategy_sel = StrategySelector(parameter_store=self.param_store)
        self.side_sel     = SideSelector()

    def route(
        self,
        market_data: Dict[str, pd.DataFrame],
        news_blocked: bool = False,
    ) -> List[TradeProposal]:
        """
        Run the full routing pipeline for all assets.

        Steps per asset:
            1. Detect regime per asset independently
            2. Check tradability (session/liquidity/news)
            3. Score asset suitability for its regime via AssetSelector
            4. Select side(s) from regime bias + indicators
            5. Select best strategy for regime
            6. Generate signal with optimised parameters
            7. Rank all proposals by composite score

        Args:
            market_data: {symbol: df} with OHLCV + features
            news_blocked: whether high-impact news is active

        Returns:
            List of TradeProposal objects sorted by composite score (may be empty)
        """
        proposals = []
        asset_scores: Dict[str, float] = {}

        # --- Phase 1: detect regimes and filter each asset ---
        asset_regimes: Dict[str, dict] = {}
        for symbol, df in market_data.items():
            if df.empty:
                continue

            try:
                regime_result = self.regime_svc.latest(
                    df=df,
                    asset=symbol,
                    news_blocked=news_blocked,
                )
            except Exception as exc:
                log.warning("Regime detection failed for %s: %s", symbol, exc)
                continue

            regime     = regime_result.smoothed_regime or regime_result.predicted_regime
            confidence = regime_result.confidence

            if not regime_result.is_tradable:
                log.info("Skipping %s -- regime '%s' is not tradable", symbol, regime)
                continue

            last_row_idx = df.index[-1:]
            can_trade = self.tf_filter.is_tradable(df.loc[last_row_idx])
            if not can_trade.iloc[0]:
                log.info("Skipping %s -- filter blocked at %s", symbol, df.index[-1])
                continue

            asset_regimes[symbol] = {
                "regime": regime,
                "confidence": confidence,
                "regime_proba": regime_result.regime_probabilities,
            }

        if not asset_regimes:
            log.info("Router: no tradable assets after filtering")
            return []

        # --- Phase 2: rank assets by regime suitability ---
        # Score each asset against its own detected regime
        for symbol, info in asset_regimes.items():
            ranked = self.asset_sel.rank(
                regime=info["regime"],
                market_data={symbol: market_data[symbol]},
                regime_confidence=info["confidence"],
            )
            score = ranked[0][1] if ranked else 0.0
            asset_scores[symbol] = max(score, 0.1)

        log.info(
            "Asset scores: %s",
            [(s, round(sc, 2)) for s, sc in sorted(
                asset_scores.items(), key=lambda x: x[1], reverse=True
            )],
        )

        # --- Phase 3: generate proposals for all qualifying assets ---
        for symbol, info in asset_regimes.items():
            df         = market_data[symbol]
            regime     = info["regime"]
            confidence = info["confidence"]

            sides = self.side_sel.select(regime, df, info["regime_proba"])

            # Breakout strategies have their own HTF trend filter, so
            # always evaluate both sides — the side selector's short-term
            # indicator bias can suppress valid breakout directions
            # (e.g. 2026-03-18: ETHUSD short breakout missed because
            # EMA/RSI still showed long bias before the crash).
            strategy_peek = self.strategy_sel.select(regime, symbol, "long")
            if strategy_peek == "session_breakout":
                sides = ["long", "short"]

            for side in sides:
                strategy_name = self.strategy_sel.select(regime, symbol, side)
                if strategy_name is None:
                    continue

                params = self.param_store.get(symbol, regime, strategy_name, side)
                strategy = get_strategy(
                    strategy_name,
                    params_long=params  if side == "long"  else None,
                    params_short=params if side == "short" else None,
                )

                try:
                    proposal = strategy.get_latest_proposal(
                        df=df,
                        side=side,
                        asset=symbol,
                        regime=regime,
                        regime_confidence=confidence,
                    )
                except Exception as exc:
                    log.warning(
                        "Signal generation failed for %s/%s/%s: %s",
                        symbol, strategy_name, side, exc,
                    )
                    continue

                if proposal is not None and proposal.is_valid():
                    proposals.append(proposal)
                    log.info(
                        "Signal: %s %s %s | entry=%.5f sl=%.5f tp=%.5f (RR=%.2f) | "
                        "regime=%s(%.0f%%) strength=%.2f asset_score=%.2f",
                        symbol, side.upper(), strategy_name,
                        proposal.entry, proposal.stop_loss, proposal.take_profit,
                        proposal.rr_ratio, regime, confidence * 100,
                        proposal.signal_strength,
                        asset_scores.get(symbol, 0),
                    )

        # --- Phase 4: sort by composite score ---
        # composite = asset_suitability * signal_strength * regime_confidence
        proposals.sort(
            key=lambda p: (
                asset_scores.get(p.asset, 0.1)
                * p.signal_strength
                * p.confidence
            ),
            reverse=True,
        )

        log.info("Router produced %d proposals across %d assets",
                 len(proposals), len(set(p.asset for p in proposals)))
        return proposals
