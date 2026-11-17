from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PortfolioConstraints:
    """Hard-coded quantitative constraints for a $5,000 options portfolio."""

    base_account_value: float = 5000.0
    min_cash_reserve_pct: float = 0.40
    target_cash_reserve_pct: float = 0.50
    panic_cash_reserve_pct: float = 0.60

    # PMCC / Diagonal allocation
    pmcc_min_allocation: float = 1000.0
    pmcc_max_allocation: float = 1300.0
    pmcc_max_contracts: int = 1
    pmcc_max_ivr: float = 30.0
    pmcc_max_vix: float = 18.0
    pmcc_long_min_dte: int = 200
    pmcc_long_max_dte: int = 300
    pmcc_long_min_delta: float = 0.80
    pmcc_long_max_delta: float = 0.85
    pmcc_short_min_dte: int = 30
    pmcc_short_max_dte: int = 45
    pmcc_short_min_delta: float = 0.25
    pmcc_short_max_delta: float = 0.30

    # Vertical Credit Spreads
    max_active_vertical_spreads: int = 3
    vertical_min_dte: int = 30
    vertical_max_dte: int = 45
    vertical_short_min_delta: float = 0.20
    vertical_short_max_delta: float = 0.28
    vertical_default_width: float = 5.0
    vertical_narrow_width: float = 2.5
    min_credit_to_width_ratio: float = 0.31  # ~1/3 width rule ($1.55-$1.65 on $5 wide)
    strict_one_third_ratio: float = 1.0 / 3.0
    min_expected_50pct_profit_usd: float = 100.00
    max_bid_ask_spread_usd: float = 0.15

    # Daily circuit breaker
    max_daily_loss_pct: float = 0.06
    max_daily_loss_usd: float = 300.00


PORTFOLIO_CONSTRAINTS = PortfolioConstraints()


@dataclass(frozen=True)
class VerticalSizingResult:
    contracts: int
    credit_per_contract: float
    spread_width: float
    total_credit_usd: float
    expected_50pct_profit_usd: float
    collateral_locked_usd: float
    target_close_price: float
    min_credit_floor: float
    meets_one_third_width: bool
    meets_profit_floor: bool


def calculate_expected_50pct_profit(credit_per_contract: float, contracts: int) -> float:
    """
    Calculate expected dollar profit at the 50% take-profit target:
    Expected_50pct_Profit = 0.50 * Credit_Per_Contract * 100 * Contracts
    """
    if credit_per_contract <= 0 or contracts <= 0:
        return 0.0
    return round(0.50 * credit_per_contract * 100.0 * contracts, 2)


def scale_vertical_contracts(
    credit_per_contract: float,
    spread_width: float = 5.0,
    min_profit_target_usd: float = PORTFOLIO_CONSTRAINTS.min_expected_50pct_profit_usd,
    account_value: float = PORTFOLIO_CONSTRAINTS.base_account_value,
    current_cash: float = 3000.0,
    panic_regime: bool = False,
) -> VerticalSizingResult:
    """
    Automatically scale Contracts so that Expected_50pct_Profit >= $100.00
    while respecting the 1/3 width rule and cash reserve floor.
    Never size a vertical spread at 1 contract if its 50% target is under $100.
    """
    if credit_per_contract <= 0:
        raise ValueError("credit_per_contract must be positive")
    if spread_width <= credit_per_contract:
        raise ValueError("spread_width must exceed credit_per_contract")

    profit_per_contract_at_50pct = 0.50 * credit_per_contract * 100.0
    required_contracts = max(1, math.ceil(min_profit_target_usd / profit_per_contract_at_50pct))

    # Check 1/3 width rule (accept >= $1.55 on $5 width as specified in Section 3.3)
    meets_one_third = (credit_per_contract / spread_width) >= PORTFOLIO_CONSTRAINTS.min_credit_to_width_ratio

    total_credit = round(credit_per_contract * 100.0 * required_contracts, 2)
    expected_50pct_profit = calculate_expected_50pct_profit(credit_per_contract, required_contracts)
    collateral_per_contract = (spread_width - credit_per_contract) * 100.0
    collateral_locked = round(collateral_per_contract * required_contracts, 2)

    reserve_floor_pct = (
        PORTFOLIO_CONSTRAINTS.panic_cash_reserve_pct
        if panic_regime
        else PORTFOLIO_CONSTRAINTS.min_cash_reserve_pct
    )
    cash_floor_usd = account_value * reserve_floor_pct
    post_trade_cash = current_cash - collateral_locked

    if post_trade_cash < cash_floor_usd:
        raise ValueError(
            f"Insufficient idle cash above reserve floor: post_trade_cash=${post_trade_cash:.2f} "
            f"< floor=${cash_floor_usd:.2f}"
        )

    target_close_price = round(credit_per_contract * 0.50, 2)
    min_credit_floor = round(min_profit_target_usd / (0.50 * 100.0 * required_contracts), 2)

    return VerticalSizingResult(
        contracts=required_contracts,
        credit_per_contract=round(credit_per_contract, 2),
        spread_width=spread_width,
        total_credit_usd=total_credit,
        expected_50pct_profit_usd=expected_50pct_profit,
        collateral_locked_usd=collateral_locked,
        target_close_price=target_close_price,
        min_credit_floor=min_credit_floor,
        meets_one_third_width=meets_one_third,
        meets_profit_floor=expected_50pct_profit >= min_profit_target_usd,
    )


def validate_pmcc_golden_rule(
    long_strike: float,
    short_strike: float,
    long_debit: float,
    short_credit: float,
    iv_rank: float,
    vix_level: float,
) -> dict[str, float | bool]:
    """
    Validate Poor Man's Covered Call (PMCC) diagonal entry constraints:
    - Net debit paid < (short_strike - long_strike) to guarantee zero upside assignment loss.
    - Allocation between $1,000 and $1,300 (or <= $1,300 cap).
    - IV Rank < 30% and VIX < 18.
    """
    net_debit = round(long_debit - short_credit, 2)
    strike_width = round(short_strike - long_strike, 2)
    if strike_width <= 0:
        raise ValueError("short_strike must be greater than long_strike for PMCC")

    golden_rule_passed = net_debit < strike_width
    if not golden_rule_passed:
        raise AssertionError(
            f"PMCC Golden Rule violated: net_debit (${net_debit:.2f}) >= strike_width (${strike_width:.2f})"
        )

    total_cost_usd = round(net_debit * 100.0, 2)
    regime_valid = (
        iv_rank < PORTFOLIO_CONSTRAINTS.pmcc_max_ivr
        and vix_level < PORTFOLIO_CONSTRAINTS.pmcc_max_vix
    )
    allocation_valid = (
        PORTFOLIO_CONSTRAINTS.pmcc_min_allocation
        <= total_cost_usd
        <= PORTFOLIO_CONSTRAINTS.pmcc_max_allocation
    )

    return {
        "net_debit": net_debit,
        "strike_width": strike_width,
        "total_cost_usd": total_cost_usd,
        "upside_guaranteed_profit_usd": round((strike_width - net_debit) * 100.0, 2),
        "golden_rule_passed": golden_rule_passed,
        "regime_valid": regime_valid,
        "allocation_valid": allocation_valid,
    }
