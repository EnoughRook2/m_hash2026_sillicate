"""
Configuration for the human-in-the-loop review layer.

Every threshold in here is a configurable heuristic tuned for THIS
simulation/demo. None of it is validated real-world underwriting policy -
treat these as adjustable starting points, not ground truth, and revisit
them with domain experts before anything resembling production use.

Keeping thresholds in one dataclass (rather than scattered through
review_flags.py / human_review.py / review.py) is the whole point: change
a number once, here, instead of hunting through multiple files.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ReviewConfig:
    # --- Q-value confidence heuristic ---
    # Confidence is a heuristic derived ONLY from Q-value separation
    # (margin between best and second-best action) and state familiarity.
    # It is NOT a calibrated probability of correctness.
    low_confidence_margin: float = 0.4
    high_confidence_margin: float = 1.0

    # --- borrower-state triggers (thresholds on the 8-dim observation) ---
    high_dti: float = 0.5
    low_buffer_days_scaled: float = 0.35
    recent_missed_threshold: float = 0.2       # consecutive_missed scaled; >0.2 == 1+ miss
    high_arrears_ratio: float = 0.5
    low_months_remaining_frac: float = 0.15
    seasonal_dip: float = -0.15
    declining_momentum: float = -0.02
    high_income_cv: float = 0.35

    # "unusual state combination": count of simultaneous stress signals
    # (from the list above) that counts as an atypical, compounding-risk
    # combination worth a human's eyes. This is a simple threshold count,
    # not a learned outlier/anomaly detector.
    unusual_combination_signal_count: int = 4

    # "serious distress + maintain": stress-signal count that, combined with
    # a `maintain` recommendation, is worth flagging as a possible
    # under-reaction by the policy.
    serious_distress_signal_count: int = 3

    # --- policy-behavior triggers ---
    # Flag when a borrower is already at (or over) their restructuring cap.
    at_restructure_cap: bool = True

    # Flag when `moratorium` has been recommended at least this many times
    # within the trailing window of months, for the same borrower.
    repeated_moratorium_window: int = 3
    repeated_moratorium_count: int = 3

    # --- state familiarity ---
    # The trained agent's q_table.pkl is a plain {state: Q-values} dict with
    # no visit counts. So "familiarity" here can only ever be a BINARY
    # seen/unseen signal (was this exact discretized state ever updated
    # during training?), not a true visit-count-based confidence score.
    # Flag recommendations coming from states the policy never trained on.
    flag_unseen_states: bool = True

    # --- diagnostics ---
    # Matches the README's existing reward-hacking warning: if one action
    # dominates decisions above this share, that's a signal to go inspect
    # the policy, not evidence that the policy is doing well.
    excessive_moratorium_diagnostic_pct: float = 0.70
