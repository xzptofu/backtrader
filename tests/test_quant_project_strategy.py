from __future__ import annotations

import unittest

from quant_project.config import StrategyConfig
from quant_project.strategy import (
    MeanReversionStrategy,
    MovingAverageCrossStrategy,
    MultiFactorAlphaStrategy,
    strategy_from_config,
)


class StrategyFactoryTestCase(unittest.TestCase):
    def test_moving_average_strategy_selection(self) -> None:
        cfg = StrategyConfig(name="moving_average_cross", short_window=12, long_window=30)
        strategy_cls, params = strategy_from_config(cfg)
        self.assertIs(strategy_cls, MovingAverageCrossStrategy)
        self.assertEqual(params["short_window"], 12)
        self.assertEqual(params["long_window"], 30)
        self.assertIn("position_size", params)

    def test_mean_reversion_strategy_selection(self) -> None:
        cfg = StrategyConfig(
            name="mean_reversion",
            mean_reversion_window=18,
            mean_reversion_entry_z=1.5,
            mean_reversion_exit_z=0.4,
            allow_short=True,
            position_size=0.6,
        )
        strategy_cls, params = strategy_from_config(cfg)
        self.assertIs(strategy_cls, MeanReversionStrategy)
        self.assertEqual(params["lookback"], 18)
        self.assertEqual(params["entry_z"], 1.5)
        self.assertEqual(params["exit_z"], 0.4)
        self.assertTrue(params["allow_short"])
        self.assertAlmostEqual(params["position_size"], 0.6)

    def test_multi_factor_strategy_selection(self) -> None:
        cfg = StrategyConfig(
            name="multi_factor_alpha",
            momentum_window=50,
            mean_reversion_window=24,
            volatility_window=15,
            momentum_weight=0.4,
            mean_reversion_weight=0.4,
            volatility_weight=0.2,
            signal_threshold=0.02,
            rebalance_interval=3,
            position_size=0.8,
            volatility_target=0.015,
            allow_short=True,
        )
        strategy_cls, params = strategy_from_config(cfg)
        self.assertIs(strategy_cls, MultiFactorAlphaStrategy)
        self.assertEqual(params["momentum_window"], 50)
        self.assertEqual(params["mean_reversion_window"], 24)
        self.assertEqual(params["volatility_window"], 15)
        self.assertEqual(params["rebalance_interval"], 3)
        self.assertAlmostEqual(params["position_size"], 0.8)
        self.assertTrue(params["allow_short"])
        self.assertAlmostEqual(params["volatility_target"], 0.015)

    def test_strategy_name_normalisation(self) -> None:
        cfg = StrategyConfig(name="Multi-Factor-Alpha")
        strategy_cls, _ = strategy_from_config(cfg)
        self.assertIs(strategy_cls, MultiFactorAlphaStrategy)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
