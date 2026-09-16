"""
Human-review trigger flags.

Every threshold used here lives in review_config.ReviewConfig - nothing is
hardcoded in this file. Each flag is a small, independent check; a decision
can trigger any number of them at once, and all triggered flags are kept
(not just the first match), so the audit log and the reviewer see the full
picture.
"""

from dataclasses import dataclass

from microfinance_env import ACTIONS
from review_config import ReviewConfig
from rl_inspection import RLInspection


@dataclass(frozen=True)
class ReviewFlag:
    code: str
    message: str


def evaluate_flags(obs, inspection: RLInspection, borrower_context: dict,
                    recent_recommendations: list, config: ReviewConfig) -> list:
    """
    obs: the 8-dim observation for this decision.
    inspection: RLInspection for this decision (see rl_inspection.py).
    borrower_context: dict with at least 'restructure_count', 'max_restructures'.
    recent_recommendations: list of this borrower's PRIOR recommended-action
        names (oldest -> newest), not including the current one.
    """
    income_cv, dti, buffer_days_scaled, seasonal_deviation, momentum, \
        arrears_ratio, consecutive_missed, months_remaining_frac = obs

    flags = []

    if inspection.q_margin < config.low_confidence_margin:
        flags.append(ReviewFlag(
            "low_confidence",
            f"Low confidence: Q-margin {inspection.q_margin:.2f} is below "
            f"the configured threshold {config.low_confidence_margin:.2f}.",
        ))

    if dti > config.high_dti:
        flags.append(ReviewFlag("high_dti", f"High DTI ({dti:.2f} > {config.high_dti:.2f})."))

    if buffer_days_scaled < config.low_buffer_days_scaled:
        flags.append(ReviewFlag(
            "low_buffer",
            f"Low savings buffer ({buffer_days_scaled:.2f} < {config.low_buffer_days_scaled:.2f}).",
        ))

    if consecutive_missed > config.recent_missed_threshold:
        flags.append(ReviewFlag(
            "recent_missed_payments",
            f"Recent missed payment(s) (consecutive_missed={consecutive_missed:.2f}).",
        ))

    if arrears_ratio > config.high_arrears_ratio:
        flags.append(ReviewFlag(
            "high_arrears",
            f"Arrears ratio {arrears_ratio:.2f} exceeds threshold {config.high_arrears_ratio:.2f}.",
        ))

    if months_remaining_frac < config.low_months_remaining_frac:
        flags.append(ReviewFlag(
            "low_remaining_tenure",
            f"Only {months_remaining_frac:.2f} of original tenure remains.",
        ))

    if (inspection.recommended_action == "moratorium"
            and months_remaining_frac < config.low_months_remaining_frac):
        flags.append(ReviewFlag(
            "moratorium_near_term_end",
            "Moratorium recommended very close to loan term end.",
        ))

    restructure_count = borrower_context.get("restructure_count", 0)
    max_restructures = borrower_context.get("max_restructures", 6)
    if config.at_restructure_cap and restructure_count >= max_restructures:
        flags.append(ReviewFlag(
            "excessive_restructuring",
            f"Borrower already at the restructuring cap ({restructure_count}/{max_restructures}).",
        ))

    window = recent_recommendations[-config.repeated_moratorium_window:]
    if window.count("moratorium") + (inspection.recommended_action == "moratorium") \
            >= config.repeated_moratorium_count:
        flags.append(ReviewFlag(
            "repeated_moratorium",
            f"Moratorium recommended in at least {config.repeated_moratorium_count} of the "
            f"last {config.repeated_moratorium_window + 1} months for this borrower.",
        ))

    stress_signal_count = len(inspection.state_signals)
    if stress_signal_count >= config.unusual_combination_signal_count:
        flags.append(ReviewFlag(
            "unusual_state_combination",
            f"{stress_signal_count} simultaneous stress signals present - an atypical, "
            f"compounding-risk combination (threshold: {config.unusual_combination_signal_count}).",
        ))

    if (inspection.recommended_action == "maintain"
            and stress_signal_count >= config.serious_distress_signal_count):
        flags.append(ReviewFlag(
            "serious_distress_with_maintain",
            f"{stress_signal_count} stress signals present but the recommendation is "
            f"'maintain' - worth a second look.",
        ))

    if config.flag_unseen_states and not inspection.state_seen_in_training:
        flags.append(ReviewFlag(
            "unseen_state",
            "This exact discretized state was never visited during training "
            "(Q-values are the untouched default) - treat the recommendation "
            "as untested, not merely low-confidence.",
        ))

    return flags
