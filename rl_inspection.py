"""
RL-recommendation inspection.

Everything here reads directly off the trained agent's Q-table and the
current discretized state. It does NOT use the rule-based baseline at all -
`baseline_policy.py`'s reasons explain the *baseline's* decision, not the
RL agent's, and mixing the two was the evaluator's existing flaw (see
README). This module only ever describes what the Q-table itself contains.

Two things this module deliberately does NOT do:
  - It does not claim "feature X caused action Y". A Q-value is a scalar
    the table happens to store for (state, action); nothing here traces
    which feature drove it. We only report which state signals are
    *present*, phrased as "the strongest signals in the state are ...",
    never as "... caused the recommendation".
  - It does not report a calibrated probability. `confidence_label` is a
    heuristic bucket from Q-value separation and whether the state was
    ever visited during training - nothing more.
"""

from dataclasses import dataclass, field

from microfinance_env import ACTIONS
from q_learning_agent import QLearningAgent, discretize
from review_config import ReviewConfig

FEATURE_NAMES = [
    "income_cv", "dti", "buffer_days_scaled", "seasonal_deviation",
    "momentum", "arrears_ratio", "consecutive_missed", "months_remaining_frac",
]


@dataclass(frozen=True)
class RLInspection:
    discretized_state: tuple
    q_values: dict                 # {"maintain": .., "reduce_extend": .., "moratorium": ..}
    recommended_action: str
    best_q: float
    second_best_q: float
    q_margin: float
    confidence_label: str          # "LOW" | "MEDIUM" | "HIGH" (heuristic, not calibrated)
    state_seen_in_training: bool   # False => state's Q-values are the untouched all-zero default
    state_signals: list = field(default_factory=list)   # e.g. ["high DTI (0.68)", ...]
    explanation: str = ""


def _confidence_label(margin: float, config: ReviewConfig) -> str:
    if margin < config.low_confidence_margin:
        return "LOW"
    if margin < config.high_confidence_margin:
        return "MEDIUM"
    return "HIGH"


def _state_signals(obs, config: ReviewConfig) -> list:
    """Describe which risk-relevant signals are present in the raw observation.

    Fixed, symmetric thresholds applied to the observation itself - this is
    intentionally the same kind of check a human reviewer could do by eye,
    not something derived from the Q-table.
    """
    income_cv, dti, buffer_days_scaled, seasonal_deviation, momentum, \
        arrears_ratio, consecutive_missed, months_remaining_frac = obs

    signals = []
    if dti > config.high_dti:
        signals.append(f"high DTI ({dti:.2f})")
    if buffer_days_scaled < config.low_buffer_days_scaled:
        signals.append(f"thin savings buffer ({buffer_days_scaled:.2f} scaled)")
    if momentum < config.declining_momentum:
        signals.append(f"declining income momentum ({momentum:.3f})")
    if seasonal_deviation < config.seasonal_dip:
        signals.append(f"income below seasonal expectation ({seasonal_deviation:.2f})")
    if arrears_ratio > 0.3:
        signals.append(f"elevated arrears ratio ({arrears_ratio:.2f})")
    if consecutive_missed > config.recent_missed_threshold:
        signals.append(f"recent missed payment(s) ({consecutive_missed:.2f} scaled)")
    if income_cv > config.high_income_cv:
        signals.append(f"high income volatility ({income_cv:.2f})")
    if months_remaining_frac < config.low_months_remaining_frac:
        signals.append(f"loan nearing term end ({months_remaining_frac:.2f} remaining)")
    return signals


def _explain(recommended_action, best_q, second_best_q, q_margin, signals) -> str:
    lines = [
        f"The Q-table assigns the highest estimated value to {recommended_action} "
        f"for this state (Q={best_q:.2f} vs. next-best Q={second_best_q:.2f}, "
        f"margin={q_margin:.2f}).",
    ]
    if signals:
        lines.append("The strongest signals present in the current state are: "
                      + "; ".join(signals) + ".")
    else:
        lines.append("No individual risk signal in the current state crosses its "
                      "configured threshold.")
    lines.append("This describes what the Q-table prefers and what is present in "
                  "the state - it does not establish that any single signal caused "
                  "the recommendation.")
    return " ".join(lines)


def inspect(agent: QLearningAgent, obs, config: ReviewConfig) -> RLInspection:
    """Build a full, honest inspection of the agent's recommendation for `obs`."""
    state = discretize(obs)

    # Membership check WITHOUT touching the defaultdict (agent.q_table[state]
    # would silently insert a zero row via __missing__ and corrupt the
    # seen/unseen signal for every state checked here but never trained on).
    state_seen = state in agent.q_table
    raw_q = agent.q_table[state]  # safe to read now; inserts zeros only if unseen, which is correct
    q_values = {ACTIONS[i]: float(raw_q[i]) for i in range(len(ACTIONS))}

    ranked = sorted(q_values.items(), key=lambda kv: kv[1], reverse=True)
    recommended_action, best_q = ranked[0]
    second_best_q = ranked[1][1]
    q_margin = best_q - second_best_q

    confidence_label = _confidence_label(q_margin, config)
    signals = _state_signals(obs, config)
    explanation = _explain(recommended_action, best_q, second_best_q, q_margin, signals)

    return RLInspection(
        discretized_state=state,
        q_values=q_values,
        recommended_action=recommended_action,
        best_q=best_q,
        second_best_q=second_best_q,
        q_margin=q_margin,
        confidence_label=confidence_label,
        state_seen_in_training=state_seen,
        state_signals=signals,
        explanation=explanation,
    )
