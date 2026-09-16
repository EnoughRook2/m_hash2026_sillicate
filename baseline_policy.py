"""
Rule-based baseline policy.

This exists for two reasons:
  1. It's your fallback demo if RL training doesn't converge cleanly in time.
  2. It's the benchmark that proves the RL agent is actually adding value -
     "our RL policy beats a sensible rule-based system on X% of borrowers"
     is a much stronger claim than showing the RL policy in isolation.

Observation indices (see microfinance_env.py):
  0 income_cv, 1 dti, 2 buffer_days_scaled, 3 seasonal_deviation,
  4 momentum, 5 arrears_ratio, 6 consecutive_missed, 7 months_remaining_frac
"""

ACTION_MAINTAIN, ACTION_REDUCE_EXTEND, ACTION_MORATORIUM = 0, 1, 2


def rule_based_policy(obs):
    income_cv, dti, buffer_days_scaled, seasonal_deviation, momentum, \
        arrears_ratio, consecutive_missed, months_remaining_frac = obs

    reasons = []

    # Genuine distress: buffer is thin AND income is currently below its
    # seasonally-expected level AND the trend isn't already recovering.
    seasonal_dip = seasonal_deviation < -0.15
    low_buffer = buffer_days_scaled < 0.35
    declining = momentum < -0.02
    high_dti = dti > 0.5
    already_missing = consecutive_missed > 0.2  # >1 missed payment (scaled by /5)

    if low_buffer and seasonal_dip and not declining:
        reasons.append("thin buffer + seasonal dip that isn't part of a longer decline")
        return ACTION_MORATORIUM, reasons

    if declining and high_dti:
        reasons.append("declining income trend combined with high debt burden")
        return ACTION_REDUCE_EXTEND, reasons

    if already_missing and arrears_ratio > 0.3:
        reasons.append("repeated missed payments and rising arrears")
        return ACTION_REDUCE_EXTEND, reasons

    if low_buffer and not seasonal_dip and not declining:
        # Buffer is low but income isn't actually depressed - a one-off wobble,
        # not worth restructuring for.
        reasons.append("low buffer but income at/above seasonal expectation - no action needed")
        return ACTION_MAINTAIN, reasons

    reasons.append("no stress signals crossed thresholds")
    return ACTION_MAINTAIN, reasons
