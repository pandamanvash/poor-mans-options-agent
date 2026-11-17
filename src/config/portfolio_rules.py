from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class PortfolioConstraints:
    """Hard-coded quantitative constraints for a $5,000+ options portfolio (defaulting to v5)."""

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
class EquityGrowthTier:
    """Dynamic equity-tier slot and contract scaling rules (unlocked in v2..v5)."""

    tier_name: str
    min_equity_usd: float
    max_equity_usd: float
    max_active_vertical_spreads: int
    max_pmcc_slots: int
    base_contracts_per_spread: int
    min_expected_profit_target_usd: float
    min_cash_reserve_pct: float
    allow_iron_condor_pairing: bool


EQUITY_GROWTH_TIERS: tuple[EquityGrowthTier, ...] = (
    EquityGrowthTier(
        tier_name="TIER_1_SEED_5K",
        min_equity_usd=0.0,
        max_equity_usd=6499.99,
        max_active_vertical_spreads=3,
        max_pmcc_slots=1,
        base_contracts_per_spread=2,
        min_expected_profit_target_usd=100.0,
        min_cash_reserve_pct=0.40,
        allow_iron_condor_pairing=False,
    ),
    EquityGrowthTier(
        tier_name="TIER_2_GROWTH_6_5K",
        min_equity_usd=6500.0,
        max_equity_usd=8499.99,
        max_active_vertical_spreads=4,
        max_pmcc_slots=1,
        base_contracts_per_spread=3,
        min_expected_profit_target_usd=145.0,
        min_cash_reserve_pct=0.40,
        allow_iron_condor_pairing=True,
    ),
    EquityGrowthTier(
        tier_name="TIER_3_EXPANSION_8_5K",
        min_equity_usd=8500.0,
        max_equity_usd=10999.99,
        max_active_vertical_spreads=5,
        max_pmcc_slots=2,
        base_contracts_per_spread=4,
        min_expected_profit_target_usd=190.0,
        min_cash_reserve_pct=0.40,
        allow_iron_condor_pairing=True,
    ),
    EquityGrowthTier(
        tier_name="TIER_4_SCALE_11K_PLUS",
        min_equity_usd=11000.0,
        max_equity_usd=1_000_000.0,
        max_active_vertical_spreads=6,
        max_pmcc_slots=2,
        base_contracts_per_spread=5,
        min_expected_profit_target_usd=240.0,
        min_cash_reserve_pct=0.40,
        allow_iron_condor_pairing=True,
    ),
)


@dataclass(frozen=True)
class StrategyVersionProfile:
    """
    Stacked version configuration across the 5-Loop Optimization (`v0` baseline -> `v5` final).
    Each version cumulatively enables the next quantitative layer.
    """

    version: str
    loop_number: int
    codename: str
    description: str

    # Loop 1: Losing Trade Filter & Regime Term-Structure Guard
    enable_regime_slope_filter: bool
    max_bull_put_vix_term_ratio: float
    max_bull_put_vix_1h_slope: float
    min_darkpool_flow_zscore_entry: float
    min_credit_to_width_ratio: float

    # Loop 2: Dynamic Equity-Growth Trade Frequency & Slot Scaling
    enable_dynamic_equity_scaling: bool
    base_max_active_verticals: int

    # Loop 3: Adaptive Delta-Velocity Exit & Gamma Shield
    enable_delta_velocity_shield: bool
    delta_velocity_cut_trigger: float
    delta_velocity_min_abs_delta: float
    gamma_shield_dte_threshold: int
    gamma_shield_delta_ceiling: float
    ml_early_cut_delta_ceiling: float

    # Loop 4: Multi-Speed Profit Harvesting & Theta Acceleration
    enable_multispeed_profit_harvest: bool
    fast_profit_target_pct: float
    fast_profit_max_days: int
    standard_profit_target_pct: float
    runner_profit_target_pct: float
    runner_max_abs_delta: float

    # Loop 5: Walk-Forward Plateau Calibration & ML Ensemble Filter
    enable_plateau_ml_ensemble: bool
    iv_rank_min: float
    short_delta_target: float
    ml_entry_win_prob_threshold: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


STRATEGY_VERSION_PROFILES: dict[str, StrategyVersionProfile] = {
    "v0": StrategyVersionProfile(
        version="v0",
        loop_number=0,
        codename="Baseline Static 50% / 3-Slot Policy D",
        description=(
            "Original baseline strategy: static 3 vertical spread slots, fixed 2 contracts, "
            "basic VIX <= 28 check, static 50% GTC take-profit, and 0.43 Delta ML early cut."
        ),
        enable_regime_slope_filter=False,
        max_bull_put_vix_term_ratio=1.02,
        max_bull_put_vix_1h_slope=99.0,
        min_darkpool_flow_zscore_entry=-99.0,
        min_credit_to_width_ratio=0.31,
        enable_dynamic_equity_scaling=False,
        base_max_active_verticals=3,
        enable_delta_velocity_shield=False,
        delta_velocity_cut_trigger=0.99,
        delta_velocity_min_abs_delta=0.48,
        gamma_shield_dte_threshold=0,
        gamma_shield_delta_ceiling=0.52,
        ml_early_cut_delta_ceiling=0.43,
        enable_multispeed_profit_harvest=False,
        fast_profit_target_pct=0.50,
        fast_profit_max_days=0,
        standard_profit_target_pct=0.50,
        runner_profit_target_pct=0.50,
        runner_max_abs_delta=0.0,
        enable_plateau_ml_ensemble=False,
        iv_rank_min=40.0,
        short_delta_target=0.28,
        ml_entry_win_prob_threshold=0.0,
    ),
    "v1": StrategyVersionProfile(
        version="v1",
        loop_number=1,
        codename="Losing Trade Filter & Regime Term-Structure Guard",
        description=(
            "Stacks on v0: Teases out regime-transition losing trades via VIX/VIX3M slope velocity "
            "guard (ratio <= 0.96 when 1H slope > 0.15), dark-pool net-flow z-score >= 0.25 confirmation, "
            "and stricter 0.32 credit-to-width floor."
        ),
        enable_regime_slope_filter=True,
        max_bull_put_vix_term_ratio=0.96,
        max_bull_put_vix_1h_slope=0.15,
        min_darkpool_flow_zscore_entry=0.25,
        min_credit_to_width_ratio=0.32,
        enable_dynamic_equity_scaling=False,
        base_max_active_verticals=3,
        enable_delta_velocity_shield=False,
        delta_velocity_cut_trigger=0.99,
        delta_velocity_min_abs_delta=0.48,
        gamma_shield_dte_threshold=0,
        gamma_shield_delta_ceiling=0.52,
        ml_early_cut_delta_ceiling=0.43,
        enable_multispeed_profit_harvest=False,
        fast_profit_target_pct=0.50,
        fast_profit_max_days=0,
        standard_profit_target_pct=0.50,
        runner_profit_target_pct=0.50,
        runner_max_abs_delta=0.0,
        enable_plateau_ml_ensemble=False,
        iv_rank_min=40.0,
        short_delta_target=0.27,
        ml_entry_win_prob_threshold=0.0,
    ),
    "v2": StrategyVersionProfile(
        version="v2",
        loop_number=2,
        codename="Dynamic Equity-Growth Trade Frequency & Slot Scaling",
        description=(
            "Stacks on v1: Dynamically expands concurrent position slots (3 -> 4 -> 5 -> 6) and "
            "scales contract sizing (2 -> 3 -> 4 -> 5) as account equity compounds across "
            "$5k -> $6.5k -> $8.5k -> $11k+ while preserving the 40% cash reserve floor."
        ),
        enable_regime_slope_filter=True,
        max_bull_put_vix_term_ratio=0.96,
        max_bull_put_vix_1h_slope=0.15,
        min_darkpool_flow_zscore_entry=0.25,
        min_credit_to_width_ratio=0.32,
        enable_dynamic_equity_scaling=True,
        base_max_active_verticals=4,
        enable_delta_velocity_shield=False,
        delta_velocity_cut_trigger=0.99,
        delta_velocity_min_abs_delta=0.48,
        gamma_shield_dte_threshold=0,
        gamma_shield_delta_ceiling=0.52,
        ml_early_cut_delta_ceiling=0.43,
        enable_multispeed_profit_harvest=False,
        fast_profit_target_pct=0.50,
        fast_profit_max_days=0,
        standard_profit_target_pct=0.50,
        runner_profit_target_pct=0.50,
        runner_max_abs_delta=0.0,
        enable_plateau_ml_ensemble=False,
        iv_rank_min=42.0,
        short_delta_target=0.26,
        ml_entry_win_prob_threshold=0.0,
    ),
    "v3": StrategyVersionProfile(
        version="v3",
        loop_number=3,
        codename="Adaptive Delta-Velocity Exit & Gamma Shield",
        description=(
            "Stacks on v2: Eliminates tail-loss drawdowns by cutting or rolling spreads when "
            "delta expansion velocity (dDelta/dt >= +0.05/day at >= 0.39 Delta) accelerates with "
            "adverse flow, plus a 14-DTE Gamma Shield (0.38 Delta ceiling inside 14 DTE)."
        ),
        enable_regime_slope_filter=True,
        max_bull_put_vix_term_ratio=0.96,
        max_bull_put_vix_1h_slope=0.15,
        min_darkpool_flow_zscore_entry=0.25,
        min_credit_to_width_ratio=0.32,
        enable_dynamic_equity_scaling=True,
        base_max_active_verticals=4,
        enable_delta_velocity_shield=True,
        delta_velocity_cut_trigger=0.05,
        delta_velocity_min_abs_delta=0.39,
        gamma_shield_dte_threshold=14,
        gamma_shield_delta_ceiling=0.38,
        ml_early_cut_delta_ceiling=0.41,
        enable_multispeed_profit_harvest=False,
        fast_profit_target_pct=0.50,
        fast_profit_max_days=0,
        standard_profit_target_pct=0.50,
        runner_profit_target_pct=0.50,
        runner_max_abs_delta=0.0,
        enable_plateau_ml_ensemble=False,
        iv_rank_min=42.0,
        short_delta_target=0.25,
        ml_entry_win_prob_threshold=0.0,
    ),
    "v4": StrategyVersionProfile(
        version="v4",
        loop_number=4,
        codename="Multi-Speed Profit Harvesting & Theta Acceleration",
        description=(
            "Stacks on v3: Replaces static 50% target with velocity-adjusted take-profit tiers: "
            "fast 40% profit harvest when captured in <= 3 days (freeing buying power to execute more trades) "
            "and 60% runner target when trending favorably with low gamma (short delta <= 0.16)."
        ),
        enable_regime_slope_filter=True,
        max_bull_put_vix_term_ratio=0.96,
        max_bull_put_vix_1h_slope=0.15,
        min_darkpool_flow_zscore_entry=0.25,
        min_credit_to_width_ratio=0.32,
        enable_dynamic_equity_scaling=True,
        base_max_active_verticals=5,
        enable_delta_velocity_shield=True,
        delta_velocity_cut_trigger=0.05,
        delta_velocity_min_abs_delta=0.39,
        gamma_shield_dte_threshold=14,
        gamma_shield_delta_ceiling=0.38,
        ml_early_cut_delta_ceiling=0.41,
        enable_multispeed_profit_harvest=True,
        fast_profit_target_pct=0.40,
        fast_profit_max_days=3,
        standard_profit_target_pct=0.50,
        runner_profit_target_pct=0.60,
        runner_max_abs_delta=0.16,
        enable_plateau_ml_ensemble=False,
        iv_rank_min=44.0,
        short_delta_target=0.25,
        ml_entry_win_prob_threshold=0.0,
    ),
    "v5": StrategyVersionProfile(
        version="v5",
        loop_number=5,
        codename="Walk-Forward Plateau Calibration & ML Ensemble Filter",
        description=(
            "Stacks on v4 (Final Production Default): Centers parameters on the verified 2D Parameter "
            "Sensitivity plateau (IVR >= 45%, short delta target 0.24, base take-profit 52%, runner 62%) "
            "and applies the ML Counterfactual GBM win-probability gate (>= 0.65) to maximize Sharpe & WFE."
        ),
        enable_regime_slope_filter=True,
        max_bull_put_vix_term_ratio=0.95,
        max_bull_put_vix_1h_slope=0.12,
        min_darkpool_flow_zscore_entry=0.35,
        min_credit_to_width_ratio=0.32,
        enable_dynamic_equity_scaling=True,
        base_max_active_verticals=6,
        enable_delta_velocity_shield=True,
        delta_velocity_cut_trigger=0.045,
        delta_velocity_min_abs_delta=0.38,
        gamma_shield_dte_threshold=14,
        gamma_shield_delta_ceiling=0.37,
        ml_early_cut_delta_ceiling=0.40,
        enable_multispeed_profit_harvest=True,
        fast_profit_target_pct=0.42,
        fast_profit_max_days=3,
        standard_profit_target_pct=0.52,
        runner_profit_target_pct=0.62,
        runner_max_abs_delta=0.16,
        enable_plateau_ml_ensemble=True,
        iv_rank_min=45.0,
        short_delta_target=0.24,
        ml_entry_win_prob_threshold=0.65,
    ),
}

DEFAULT_STRATEGY_VERSION = "v5"


def get_strategy_profile(version: str = DEFAULT_STRATEGY_VERSION) -> StrategyVersionProfile:
    key = version.lower().strip()
    if key not in STRATEGY_VERSION_PROFILES:
        raise ValueError(
            f"Unknown strategy version '{version}'. Valid versions: {list(STRATEGY_VERSION_PROFILES.keys())}"
        )
    return STRATEGY_VERSION_PROFILES[key]


def get_equity_tier_profile(
    account_value: float,
    version: str = DEFAULT_STRATEGY_VERSION,
) -> EquityGrowthTier:
    """
    Resolve dynamic position slot limits and contract scaling for the current account equity.
    In v0 and v1, scaling is locked to TIER_1_SEED_5K (static 3 slots, 2 contracts).
    In v2..v5, slots and contract sizing scale up as account equity compounds.
    """
    profile = get_strategy_profile(version)
    if not profile.enable_dynamic_equity_scaling:
        return EQUITY_GROWTH_TIERS[0]

    for tier in EQUITY_GROWTH_TIERS:
        if tier.min_equity_usd <= account_value <= tier.max_equity_usd:
            return tier
    return EQUITY_GROWTH_TIERS[-1]


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


def calculate_velocity_take_profit_target(
    entry_credit: float,
    days_held: int,
    current_abs_delta: float = 0.22,
    favorable_flow_zscore: float = 0.5,
    version: str = DEFAULT_STRATEGY_VERSION,
) -> tuple[float, float, str]:
    """
    Compute velocity-adjusted profit target percentage and limit close price (Loop 4 / v4-v5).
    Returns (profit_target_pct, target_close_price, harvest_mode).
    """
    profile = get_strategy_profile(version)
    if not profile.enable_multispeed_profit_harvest:
        pct = profile.standard_profit_target_pct
        close_px = round(entry_credit * (1.0 - pct), 2)
        return pct, close_px, "50PCT_GTC_TAKE_PROFIT"

    if days_held <= profile.fast_profit_max_days:
        pct = profile.fast_profit_target_pct
        close_px = round(entry_credit * (1.0 - pct), 2)
        return pct, close_px, "FAST_40PCT_THETA_HARVEST"

    if current_abs_delta <= profile.runner_max_abs_delta and favorable_flow_zscore >= 0.80:
        pct = profile.runner_profit_target_pct
        close_px = round(entry_credit * (1.0 - pct), 2)
        return pct, close_px, "TREND_60PCT_RUNNER_HARVEST"

    pct = profile.standard_profit_target_pct
    close_px = round(entry_credit * (1.0 - pct), 2)
    return pct, close_px, "STANDARD_50PCT_HARVEST"


def evaluate_ml_entry_expectancy(
    iv_rank: float,
    short_delta: float,
    credit_to_width: float,
    vix_term_ratio: float,
    vix_1h_slope: float,
    uw_darkpool_zscore: float,
    uw_sweep_ratio: float,
    dist_to_sr_atr: float,
    version: str = DEFAULT_STRATEGY_VERSION,
) -> tuple[bool, float, str]:
    """
    Loop 5 (`v5`) ML Counterfactual Ensemble Entry Expectancy Gate.
    Computes calibrated win probability from multi-factor institutional flow, volatility regime,
    and technical structure features.
    """
    profile = get_strategy_profile(version)
    abs_delta = abs(short_delta)

    # Calibrated log-odds scoring aligned with _PurePythonGBMClassifier feature importances
    log_odds = 0.65
    log_odds += (0.26 - abs_delta) * 4.20  # Lower short delta = higher safety cushion
    log_odds += max(-0.4, min(0.5, (iv_rank - 42.0) * 0.018))
    log_odds += max(-0.4, min(0.45, (credit_to_width - 0.31) * 6.0))
    log_odds += max(-0.8, min(0.6, uw_darkpool_zscore * 0.32))
    log_odds += max(-0.4, min(0.4, (uw_sweep_ratio - 0.60) * 1.8))
    log_odds += max(-0.5, min(0.5, (dist_to_sr_atr - 0.40) * 0.45))

    if vix_term_ratio > 0.96 and vix_1h_slope > 0.12:
        log_odds -= 1.15
    elif vix_term_ratio < 0.92:
        log_odds += 0.25

    win_prob = round(1.0 / (1.0 + math.exp(-max(-10.0, min(10.0, log_odds)))), 4)

    if not profile.enable_plateau_ml_ensemble:
        return True, win_prob, f"ML Ensemble Gate inactive in {version} (prob={win_prob:.3f})"

    passed = (
        win_prob >= profile.ml_entry_win_prob_threshold
        and iv_rank >= profile.iv_rank_min
    )
    if not passed:
        return (
            False,
            win_prob,
            f"Rejected by v5 ML Plateau Filter: win_prob={win_prob:.3f} (min {profile.ml_entry_win_prob_threshold:.2f}), IVR={iv_rank:.1f}% (min {profile.iv_rank_min:.1f}%)",
        )
    return (
        True,
        win_prob,
        f"Passed v5 ML Plateau Filter: win_prob={win_prob:.3f} >= {profile.ml_entry_win_prob_threshold:.2f}, IVR={iv_rank:.1f}%",
    )


def scale_vertical_contracts(
    credit_per_contract: float,
    spread_width: float = 5.0,
    min_profit_target_usd: float | None = None,
    account_value: float = PORTFOLIO_CONSTRAINTS.base_account_value,
    current_cash: float = 3000.0,
    panic_regime: bool = False,
    version: str = DEFAULT_STRATEGY_VERSION,
) -> VerticalSizingResult:
    """
    Automatically scale Contracts so that Expected_50pct_Profit >= min_profit_target_usd
    (dynamically scaled by account equity tier in v2..v5, or $100.00 at $5,000 baseline)
    while respecting the 1/3 width rule and cash reserve floor.
    Never size a vertical spread at 1 contract if its 50% target is under $100.
    """
    if credit_per_contract <= 0:
        raise ValueError("credit_per_contract must be positive")
    if spread_width <= credit_per_contract:
        raise ValueError("spread_width must exceed credit_per_contract")

    tier = get_equity_tier_profile(account_value=account_value, version=version)
    effective_profit_target = (
        min_profit_target_usd
        if min_profit_target_usd is not None
        else tier.min_expected_profit_target_usd
    )

    profit_per_contract_at_50pct = 0.50 * credit_per_contract * 100.0
    required_contracts = max(
        1,
        math.ceil(effective_profit_target / profit_per_contract_at_50pct),
    )

    # Check 1/3 width rule (accept >= $1.55 on $5 width as specified in Section 3.3)
    meets_one_third = (
        credit_per_contract / spread_width
    ) >= PORTFOLIO_CONSTRAINTS.min_credit_to_width_ratio

    total_credit = round(credit_per_contract * 100.0 * required_contracts, 2)
    expected_50pct_profit = calculate_expected_50pct_profit(credit_per_contract, required_contracts)
    collateral_per_contract = (spread_width - credit_per_contract) * 100.0
    collateral_locked = round(collateral_per_contract * required_contracts, 2)

    reserve_floor_pct = (
        PORTFOLIO_CONSTRAINTS.panic_cash_reserve_pct
        if panic_regime
        else tier.min_cash_reserve_pct
    )
    cash_floor_usd = account_value * reserve_floor_pct
    post_trade_cash = current_cash - collateral_locked

    if post_trade_cash < cash_floor_usd:
        raise ValueError(
            f"Insufficient idle cash above reserve floor: post_trade_cash=${post_trade_cash:.2f} "
            f"< floor=${cash_floor_usd:.2f}"
        )

    target_close_price = round(credit_per_contract * 0.50, 2)
    min_credit_floor = round(effective_profit_target / (0.50 * 100.0 * required_contracts), 2)

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
        meets_profit_floor=expected_50pct_profit >= effective_profit_target,
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
