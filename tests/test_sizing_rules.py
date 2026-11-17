from __future__ import annotations

import unittest

from src.config.portfolio_rules import (
    calculate_expected_50pct_profit,
    scale_vertical_contracts,
    validate_pmcc_golden_rule,
)


class TestSizingRules(unittest.TestCase):
    def test_vertical_spread_scales_to_2_contracts_for_100_dollar_profit_floor(self):
        single_contract_profit = calculate_expected_50pct_profit(1.60, 1)
        self.assertEqual(single_contract_profit, 80.00)

        sizing = scale_vertical_contracts(
            credit_per_contract=1.60,
            spread_width=5.0,
            account_value=5000.0,
            current_cash=3500.0,
        )
        self.assertEqual(sizing.contracts, 2)
        self.assertEqual(sizing.total_credit_usd, 320.00)
        self.assertEqual(sizing.expected_50pct_profit_usd, 160.00)
        self.assertGreaterEqual(sizing.expected_50pct_profit_usd, 100.00)
        self.assertEqual(sizing.collateral_locked_usd, 680.00)
        self.assertEqual(sizing.target_close_price, 0.80)
        self.assertTrue(sizing.meets_one_third_width)
        self.assertTrue(sizing.meets_profit_floor)

    def test_cash_reserve_floor_enforced_at_40_percent(self):
        with self.assertRaises(ValueError):
            scale_vertical_contracts(
                credit_per_contract=1.60,
                spread_width=5.0,
                account_value=5000.0,
                current_cash=2500.0,
            )

    def test_pmcc_golden_rule_assertion(self):
        valid = validate_pmcc_golden_rule(
            long_strike=230.0,
            short_strike=250.0,
            long_debit=14.20,
            short_credit=2.10,
            iv_rank=24.5,
            vix_level=16.2,
        )
        self.assertEqual(valid["net_debit"], 12.10)
        self.assertEqual(valid["strike_width"], 20.00)
        self.assertEqual(valid["total_cost_usd"], 1210.00)
        self.assertTrue(valid["golden_rule_passed"])
        self.assertTrue(valid["regime_valid"])
        self.assertTrue(valid["allocation_valid"])

        with self.assertRaises(AssertionError):
            validate_pmcc_golden_rule(
                long_strike=235.0,
                short_strike=245.0,
                long_debit=13.50,
                short_credit=1.50,
                iv_rank=22.0,
                vix_level=15.5,
            )


if __name__ == "__main__":
    unittest.main()
