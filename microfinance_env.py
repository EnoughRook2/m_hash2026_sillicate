"""
Microfinance repayment-structuring environment.

This module has two parts:
  1. BorrowerSimulator  - generates a synthetic borrower's monthly income
                          (trend + seasonality + shocks + noise) and models
                          how they respond to different repayment terms.
  2. MicrofinanceEnv    - wraps the simulator as a Gymnasium environment so
                          any RL algorithm (tabular Q-learning, DQN, PPO...)
                          can be trained on it.

Design choices are commented inline so the team can argue about / change
them quickly - this is a starting point, not a final model.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


BORROWER_TYPES = ["trader", "agricultural", "shg_group"]

# Seasonal index per calendar month (Jan..Dec), multiplicative.
# 1.0 = average month. These are illustrative, not fitted to real data.
SEASONAL_PROFILES = {
    # Traders: festive-season bump (Oct/Nov in India), lean months mid-year
    "trader":       [0.85, 0.85, 0.90, 0.95, 0.95, 0.90, 0.90, 0.95, 1.05, 1.35, 1.30, 1.10],
    # Agricultural: harvest-linked income spikes, long lean stretch before harvest
    "agricultural": [0.60, 0.55, 0.60, 0.70, 0.85, 1.10, 1.60, 1.70, 1.30, 0.90, 0.70, 0.65],
    # SHG groups: mildly seasonal, more stable (pooled/diversified income)
    "shg_group":    [0.95, 0.95, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.05, 1.10, 1.05, 1.00],
}

ACTIONS = ["maintain", "reduce_extend", "moratorium"]
N_ACTIONS = len(ACTIONS)


class BorrowerSimulator:
    """Generates one synthetic borrower and simulates month-by-month cash flow."""

    def __init__(self, rng: np.random.Generator, borrower_type: str | None = None):
        self.rng = rng
        self.borrower_type = borrower_type or rng.choice(BORROWER_TYPES)
        self.seasonal = np.array(SEASONAL_PROFILES[self.borrower_type])

        # Base monthly income, drawn from a plausible microfinance range.
        self.base_income = rng.uniform(6000, 18000)

        # Long-run trend: >1 improving borrower, <1 slowly deteriorating.
        # Most borrowers are roughly flat; a minority drift up or down.
        self.trend_rate = rng.choice(
            [rng.uniform(-0.010, -0.004), rng.uniform(-0.001, 0.001), rng.uniform(0.004, 0.010)],
            p=[0.2, 0.6, 0.2],
        )

        # Income volatility (noise on top of trend+seasonality).
        self.volatility = rng.uniform(0.05, 0.30)

        # Probability and severity of an income shock each month
        # (medical emergency, family event, local shock, etc.)
        self.shock_prob = rng.uniform(0.03, 0.12)
        self.shock_severity = rng.uniform(0.3, 0.7)  # fraction of income lost in a shock month

        # Essential monthly expenses as a fraction of *expected* income.
        self.expense_ratio = rng.uniform(0.55, 0.80)

        # Loan terms
        self.principal = rng.uniform(15000, 60000)
        self.tenure_months = int(rng.choice([12, 18, 24]))
        self.annual_rate = 0.24  # typical microfinance annual rate, illustrative
        self.original_emi = self._amortized_payment(self.principal, self.annual_rate, self.tenure_months)

        # Mutable loan/borrower state, set in reset()
        self.t = 0
        self.month0 = int(rng.integers(0, 12))
        self.outstanding_principal = self.principal
        self.months_remaining = self.tenure_months
        self.buffer = rng.uniform(0.2, 1.5) * self.base_income  # starting savings buffer
        self.arrears = 0.0
        self.consecutive_missed = 0
        self.restructure_count = 0
        self.prev_action = 0  # "maintain"
        self.income_history = []  # trailing true incomes, for CV/seasonal/momentum features
        self.expected_required_payment = self.original_emi

    @staticmethod
    def _amortized_payment(principal, annual_rate, n_months):
        r = annual_rate / 12.0
        if r == 0:
            return principal / n_months
        return principal * r / (1 - (1 + r) ** (-n_months))

    def _true_income(self, t):
        """Underlying (unobserved-by-model-until-realised) income for month t."""
        month_idx = (self.month0 + t) % 12
        seasonal_mult = self.seasonal[month_idx]
        trend_mult = (1 + self.trend_rate) ** t
        noise = self.rng.lognormal(mean=0.0, sigma=self.volatility)
        income = self.base_income * seasonal_mult * trend_mult * noise

        shocked = self.rng.random() < self.shock_prob
        if shocked:
            income *= (1 - self.shock_severity)
        return max(income, 0.0), shocked, seasonal_mult

    def required_payment(self, action_idx):
        """Payment owed this month given the chosen restructuring action."""
        action = ACTIONS[action_idx]
        if action == "maintain":
            return self._amortized_payment(self.outstanding_principal, self.annual_rate,
                                             max(self.months_remaining, 1))
        elif action == "reduce_extend":
            # Lower the payment by extending the effective tenure (NPV roughly preserved
            # by recomputing amortization over a longer horizon). Always at least 1 month.
            base_n = max(self.months_remaining, 1)
            extended_n = max(min(base_n + 6, base_n * 2 + 6), 1)
            return self._amortized_payment(self.outstanding_principal, self.annual_rate, extended_n)
        elif action == "moratorium":
            # Interest-only: principal doesn't reduce this month.
            r = self.annual_rate / 12.0
            return self.outstanding_principal * r
        else:
            raise ValueError(action)

    def step(self, action_idx):
        """Advance one month. Returns a dict of realised values for this month."""
        true_income, shocked, seasonal_mult = self._true_income(self.t)
        self.income_history.append(true_income)

        required = self.required_payment(action_idx)
        expenses = self.expense_ratio * self.base_income * (self.seasonal[(self.month0 + self.t) % 12])
        disposable = max(true_income - expenses, 0.0)

        # Borrower pays what they can from disposable income, then dips into
        # buffer (savings) to try to cover the rest, up to a limited willingness.
        shortfall = max(required - disposable, 0.0)
        buffer_draw = min(shortfall, self.buffer, required * 0.5)  # won't drain buffer in one shot
        payment_made = min(required, disposable + buffer_draw)
        self.buffer = max(self.buffer - buffer_draw, 0.0)
        self.buffer += max(disposable - payment_made, 0.0) * 0.5  # some leftover saved

        missed_amount = required - payment_made
        if missed_amount > 1e-6:
            self.arrears += missed_amount
            self.consecutive_missed += 1
        else:
            self.consecutive_missed = 0
            # arrears slowly recovered if borrower overpays relative to plan is out of scope here

        # Principal reduction only happens on "maintain"/"reduce_extend" (moratorium is interest-only)
        action = ACTIONS[action_idx]
        if action in ("maintain", "reduce_extend") and payment_made > 0:
            r = self.annual_rate / 12.0
            interest_component = self.outstanding_principal * r
            principal_component = max(payment_made - interest_component, 0.0)
            self.outstanding_principal = max(self.outstanding_principal - principal_component, 0.0)

        if action_idx != self.prev_action:
            self.restructure_count += 1
        self.prev_action = action_idx

        self.months_remaining -= 1
        self.t += 1

        realised = {
            "true_income": true_income,
            "shocked": shocked,
            "seasonal_mult": seasonal_mult,
            "required": required,
            "payment_made": payment_made,
            "missed_amount": missed_amount,
        }
        return realised


class MicrofinanceEnv(gym.Env):
    """
    Observation (8 features, all roughly normalised):
      0. income_cv          - volatility of trailing income (coefficient of variation)
      1. dti                - required payment / trailing avg income
      2. buffer_days         - savings buffer expressed in days of avg expense, log-scaled
      3. seasonal_deviation  - (last actual income / seasonally-expected income) - 1
      4. momentum            - slope of trailing income trend, normalised
      5. arrears_ratio       - cumulative arrears / original EMI
      6. consecutive_missed  - capped count of consecutive missed/partial payments
      7. months_remaining_frac - fraction of original tenure left

    Actions: 0 = maintain, 1 = reduce_extend, 2 = moratorium
    Default is NOT an action - it's a terminal state the environment detects
    when arrears become unrecoverable (3+ consecutive full misses).
    """

    metadata = {"render_modes": []}

    def __init__(self, seed=None, max_restructures=6, default_streak=3):
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Box(low=-3.0, high=10.0, shape=(8,), dtype=np.float32)
        self.max_restructures = max_restructures
        self.default_streak = default_streak
        self.borrower = None

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.borrower = BorrowerSimulator(self.rng)
        obs = self._observe(last=None)
        info = {"borrower_type": self.borrower.borrower_type}
        return obs, info

    def _observe(self, last):
        b = self.borrower
        hist = b.income_history[-6:] if b.income_history else [b.base_income]
        mean_income = float(np.mean(hist))
        std_income = float(np.std(hist))
        income_cv = std_income / mean_income if mean_income > 0 else 0.0

        required_now = b.required_payment(b.prev_action)
        dti = required_now / mean_income if mean_income > 0 else 1.0

        avg_daily_expense = (b.expense_ratio * b.base_income) / 30.0
        buffer_days = b.buffer / avg_daily_expense if avg_daily_expense > 0 else 0.0
        buffer_days_scaled = np.log1p(buffer_days) / np.log1p(60)  # ~0 at 0 days, ~1 at 60 days

        # BUG FIX: this method is called AFTER b.step() has already incremented
        # b.t, so (b.month0 + b.t) % 12 points at the UPCOMING month, not the
        # month hist[-1]'s income actually belongs to - meaning every
        # seasonal_deviation reading compared last month's real income against
        # next month's seasonal norm. Caught via hand-verifying a worked
        # example: an agricultural borrower's true income of 6370 (a real
        # calendar-month-3 figure, seasonal mult 0.70) was being compared
        # against a calendar-month-4 expectation (mult 0.85), reporting
        # deviation -0.476 instead of the correct -0.364. When hist has at
        # least one real entry, the month that income belongs to is (b.t - 1),
        # not b.t.
        if hist and len(b.income_history) > 0:
            month_idx = (b.month0 + b.t - 1) % 12
        else:
            month_idx = (b.month0 + b.t) % 12
        seasonal_expected = b.base_income * b.seasonal[month_idx]
        last_actual = hist[-1] if hist else seasonal_expected
        seasonal_deviation = (last_actual / seasonal_expected) - 1.0 if seasonal_expected > 0 else 0.0

        if len(hist) >= 3:
            x = np.arange(len(hist))
            slope = np.polyfit(x, hist, 1)[0]
            momentum = slope / mean_income if mean_income > 0 else 0.0
        else:
            momentum = 0.0

        arrears_ratio = b.arrears / b.original_emi if b.original_emi > 0 else 0.0
        consecutive_missed = min(b.consecutive_missed, 5) / 5.0
        months_remaining_frac = b.months_remaining / b.tenure_months

        obs = np.array([
            income_cv, dti, buffer_days_scaled, seasonal_deviation,
            momentum, arrears_ratio, consecutive_missed, months_remaining_frac
        ], dtype=np.float32)
        return np.clip(obs, -3.0, 10.0)

    def step(self, action_idx):
        b = self.borrower
        action_before_this_step = b.prev_action  # capture BEFORE b.step() mutates it
        realised = b.step(action_idx)

        # --- reward terms (tune these weights during Day 5 of the plan) ---
        # IMPORTANT: collection is measured against the ORIGINAL scheduled EMI,
        # not the action-adjusted "required" amount. Measuring against the
        # adjusted amount let the agent lower its own bar by picking moratorium
        # (required shrinks to interest-only) and then trivially "clear" it -
        # a real reward-hacking bug caught during testing: the trained agent
        # picked moratorium ~85% of the time regardless of borrower state.
        collection_ratio = realised["payment_made"] / b.original_emi if b.original_emi > 0 else 1.0
        w1_collection = 1.0
        w2_dti_overshoot = 2.0
        w3_restructure_change = 0.15
        w4_missed_payment = 0.6
        w5_relief_cost = 0.5  # explicit cost of granting relief, so it must be "earned" by future benefit

        obs = self._observe(realised)
        dti = obs[1]
        dti_safe_threshold = 0.5  # matches the "<=50% of income" guardrail from underwriting

        reward = w1_collection * collection_ratio
        reward -= w2_dti_overshoot * max(0.0, dti - dti_safe_threshold)
        reward -= w3_restructure_change * (1.0 if action_idx != action_before_this_step else 0.0)
        reward -= w4_missed_payment * (1.0 if realised["missed_amount"] > 1e-6 else 0.0)
        # Moratorium has a direct cost (foregone principal collection this month) so it
        # must be justified by avoiding worse outcomes later, not chosen for free.
        reward -= w5_relief_cost * (1.0 if action_idx == 2 else 0.0)

        terminated = False
        truncated = False
        info = {"realised": realised, "borrower_type": b.borrower_type}

        # These are independent conditions (not elif): churn penalty should never
        # block detecting default or loan completion - that was a real bug caught
        # in testing (episodes ran past month 0 into negative tenure and crashed
        # the amortization formula). Keep these as separate ifs.
        if b.restructure_count > self.max_restructures:
            reward -= 1.0  # discourage endless churn of restructuring

        if b.consecutive_missed >= self.default_streak:
            reward -= 10.0
            terminated = True
            info["outcome"] = "default"
        elif b.months_remaining <= 0:
            # Terminal bonus is based on how much PRINCIPAL actually got paid down,
            # not just low arrears - arrears alone doesn't catch a loan that spent
            # the whole term in moratorium and left principal largely untouched.
            principal_recovered_frac = 1.0 - (b.outstanding_principal / b.principal)
            reward += 10.0 * max(0.0, principal_recovered_frac)
            if principal_recovered_frac < 0.5:
                reward -= 8.0  # meaningful unpaid balance at term end is a bad outcome
                info["outcome"] = "term_end_unresolved"
            else:
                info["outcome"] = "repaid"
            terminated = True

        # Safety net: hard-stop any episode that somehow runs long, so a future
        # bug degrades gracefully (truncation) instead of crashing.
        if b.t > b.tenure_months + 12:
            truncated = True

        return obs, float(reward), terminated, truncated, info
