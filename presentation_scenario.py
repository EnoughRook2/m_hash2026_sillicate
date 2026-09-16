"""
Presentation / demo scenario support.

Converts manually-authored, human-readable monthly borrower statistics into
the exact same 8-dim RL observation that MicrofinanceEnv._observe() produces
during training and evaluation - by building a stand-in borrower object with
the fields _observe() actually reads, and calling _observe() directly rather
than re-deriving any feature formula here.

The one exception is `dti`: the real formula (required payment / trailing
income) depends on which restructuring action's amortization schedule is
currently active, which this module does not simulate (there is no live
loan being stepped forward in presentation mode - see presentation_demo.py).
Since the presentation input already asks for DTI as a human-readable
ratio, that value is used directly and is the one feature explicitly
overridden after calling _observe().

This module never touches q_table.pkl and never calls
QLearningAgent.update() / MicrofinanceEnv.reset() / MicrofinanceEnv.step().
It only reuses the pure observation-construction logic.
"""

from dataclasses import dataclass

import numpy as np

from microfinance_env import MicrofinanceEnv, BorrowerSimulator, BORROWER_TYPES

DEFAULT_TENURE_MONTHS = 24
DEFAULT_EXPENSE_RATIO = 0.65
DEFAULT_START_CALENDAR_MONTH = 1  # January


@dataclass(frozen=True)
class MonthlyStats:
    month: int
    income: float
    dti: float
    savings_buffer: float = 0.0
    arrears: float = 0.0
    consecutive_missed: int = 0
    months_remaining: int | None = None  # None => derived from tenure_months


@dataclass(frozen=True)
class PresentationCase:
    borrower_id: str
    borrower_type: str
    months: list  # list[MonthlyStats], ordered 1..N
    tenure_months: int = DEFAULT_TENURE_MONTHS
    original_emi: float | None = None
    typical_monthly_income: float | None = None
    expense_ratio: float = DEFAULT_EXPENSE_RATIO
    start_calendar_month: int = DEFAULT_START_CALENDAR_MONTH  # 1-12


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _build_month(i, m):
    _require("income" in m, f"month {i}: missing 'income'")
    _require(m["income"] > 0, f"month {i}: 'income' must be a positive number, got {m['income']!r}")
    _require("dti" in m, f"month {i}: missing 'dti'")
    _require(m["dti"] >= 0, f"month {i}: 'dti' must be a non-negative number, got {m['dti']!r}")

    savings_buffer = m.get("savings_buffer", 0.0)
    _require(savings_buffer >= 0, f"month {i}: 'savings_buffer' must be >= 0")
    arrears = m.get("arrears", 0.0)
    _require(arrears >= 0, f"month {i}: 'arrears' must be >= 0")
    consecutive_missed = m.get("consecutive_missed", 0)
    _require(isinstance(consecutive_missed, int) and consecutive_missed >= 0,
              f"month {i}: 'consecutive_missed' must be a non-negative int")
    months_remaining = m.get("months_remaining")
    if months_remaining is not None:
        _require(isinstance(months_remaining, int) and months_remaining >= 0,
                  f"month {i}: 'months_remaining' must be a non-negative int")

    return MonthlyStats(
        month=i, income=float(m["income"]), dti=float(m["dti"]),
        savings_buffer=float(savings_buffer), arrears=float(arrears),
        consecutive_missed=consecutive_missed, months_remaining=months_remaining,
    )


def build_case(raw: dict) -> PresentationCase:
    """Validate a raw dict (e.g. loaded from JSON) and build a PresentationCase.
    Fails fast with a specific ValueError on the first problem found - no
    silent defaulting on malformed required input."""
    _require("borrower_id" in raw, "presentation case is missing 'borrower_id'")
    _require("months" in raw and raw["months"], "presentation case has no 'months' entries")

    borrower_type = raw.get("borrower_type", "trader")
    _require(borrower_type in BORROWER_TYPES,
              f"borrower_type must be one of {BORROWER_TYPES}, got {borrower_type!r}")

    tenure_months = raw.get("tenure_months", DEFAULT_TENURE_MONTHS)
    _require(isinstance(tenure_months, int) and tenure_months > 0,
              f"tenure_months must be a positive int, got {tenure_months!r}")

    expense_ratio = raw.get("expense_ratio", DEFAULT_EXPENSE_RATIO)
    _require(0 < expense_ratio < 1, f"expense_ratio must be between 0 and 1, got {expense_ratio!r}")

    start_calendar_month = raw.get("start_calendar_month", DEFAULT_START_CALENDAR_MONTH)
    _require(1 <= start_calendar_month <= 12,
              f"start_calendar_month must be 1-12, got {start_calendar_month!r}")

    months = []
    for i, m in enumerate(raw["months"], start=1):
        _require("month" in m, f"months[{i}] is missing 'month'")
        _require(m["month"] == i,
                  f"months must be numbered sequentially from 1 with no gaps; "
                  f"expected month {i}, got {m['month']}")
        months.append(_build_month(i, m))

    original_emi = raw.get("original_emi")
    if any(m.arrears > 0 for m in months):
        _require(original_emi is not None and original_emi > 0,
                  "original_emi is required (and must be > 0) when any month has "
                  "nonzero arrears, since arrears_ratio = arrears / original_emi")

    typical_monthly_income = raw.get("typical_monthly_income")
    if typical_monthly_income is None:
        typical_monthly_income = float(np.mean([m.income for m in months]))

    return PresentationCase(
        borrower_id=raw["borrower_id"],
        borrower_type=borrower_type,
        months=months,
        tenure_months=tenure_months,
        original_emi=original_emi,
        typical_monthly_income=typical_monthly_income,
        expense_ratio=expense_ratio,
        start_calendar_month=start_calendar_month,
    )


def _make_stand_in_borrower(case: PresentationCase, month_stats: MonthlyStats,
                             income_history: list) -> BorrowerSimulator:
    """Populate a BorrowerSimulator-shaped object from manually-supplied
    stats so MicrofinanceEnv._observe() can compute the observation with its
    real formulas. Random-generation fields (trend_rate, volatility, shock_*,
    principal, ...) are never read by _observe() and are left at whatever
    __init__ draws; only fields _observe() actually touches are overwritten."""
    sim = BorrowerSimulator(rng=np.random.default_rng(0), borrower_type=case.borrower_type)

    sim.base_income = case.typical_monthly_income
    sim.expense_ratio = case.expense_ratio
    sim.tenure_months = case.tenure_months
    sim.original_emi = case.original_emi or 1.0  # arrears is validated to be 0 whenever this is unset
    sim.buffer = month_stats.savings_buffer
    sim.arrears = month_stats.arrears
    sim.consecutive_missed = month_stats.consecutive_missed
    sim.months_remaining = (
        month_stats.months_remaining if month_stats.months_remaining is not None
        else max(case.tenure_months - (month_stats.month - 1), 0)
    )
    sim.income_history = income_history
    sim.month0 = case.start_calendar_month - 1  # env uses 0-indexed calendar months
    sim.t = month_stats.month
    # prev_action only feeds required_payment() inside _observe(), whose
    # result is discarded below in favor of the directly-supplied dti - so
    # its value has no effect on the resulting observation. Left at the
    # constructor default (0 / "maintain").

    return sim


def observation_for_month(case: PresentationCase, month_stats: MonthlyStats,
                           income_history: list) -> np.ndarray:
    """Return the 8-dim observation for one manually-supplied month, computed
    with the same formulas MicrofinanceEnv uses during training/evaluation."""
    sim = _make_stand_in_borrower(case, month_stats, income_history)
    env = MicrofinanceEnv()
    env.borrower = sim
    obs = env._observe(last=None)
    obs[1] = month_stats.dti  # see module docstring: dti is supplied directly
    return np.clip(obs, -3.0, 10.0)


def iter_observations(case: PresentationCase):
    """Yield (month_stats, obs) for every month in the case, in order, with
    income_history accumulated exactly as MicrofinanceEnv would trail it."""
    income_history = []
    for month_stats in case.months:
        income_history.append(month_stats.income)
        obs = observation_for_month(case, month_stats, income_history[-6:])
        yield month_stats, obs
