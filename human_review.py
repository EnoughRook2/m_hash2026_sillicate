"""
Human-in-the-loop review layer (v2).

Sits between the trained agent's recommendation and actually executing an
action. This module owns exactly one job: decide whether a recommendation
auto-executes or needs a human, and produce a ReviewDecision record that
keeps every field the spec requires separate:

    rl_recommended_action   - what the Q-table recommends. NEVER overwritten.
    human_decision_action   - what a human chose, if anyone reviewed it (None otherwise).
    final_action            - what actually gets executed in the environment.
    override_status         - "approved" | "overridden" | "escalated" |
                               "deferred" | "needs_review".
    override_reason         - the reviewer's stated reason (empty if none given).

The RL recommendation is always recorded and never mutated by a human
decision - overriding it changes `final_action` and `override_status`, not
`rl_recommended_action`.

This is v1's replacement, not an incremental patch on top of it: v1
(`HumanReviewGate` with a single `margin_threshold` and a handful of
inline-threshold checks) is superseded by the modules this file composes:
`rl_inspection.py` (Q-table/state inspection), `review_flags.py`
(configurable triggers), `review_config.py` (thresholds), and
`audit_log.py` (persistence). v1's `review_log.csv` is left untouched as a
historical record; see audit_log.py's docstring.

Design note: this is "shadow mode" in the same sense v1 was - it does not
change how the agent was trained, and can be dropped on top of an
already-trained q_table.pkl with zero retraining.
"""

from dataclasses import dataclass, asdict
import time

from microfinance_env import ACTIONS
from q_learning_agent import QLearningAgent
from review_config import ReviewConfig
from rl_inspection import inspect, RLInspection
from review_flags import evaluate_flags
import audit_log

VALID_OVERRIDE_STATUSES = {"approved", "overridden", "escalated", "deferred", "needs_review"}


@dataclass
class ReviewDecision:
    # --- identifiers ---
    timestamp: float
    episode_id: str
    borrower_id: str
    borrower_type: str
    month: int

    # --- observation / state (as actually presented to the agent) ---
    obs: list
    discretized_state: list

    # --- RL recommendation: read-only facts about what the Q-table said ---
    rl_recommended_action: str
    q_values: dict
    q_margin: float
    confidence_label: str
    state_seen_in_training: bool
    rl_explanation: str

    # --- review triggers ---
    review_flags: list          # list of flag codes
    review_flag_details: list   # list of human-readable flag messages
    flagged: bool

    # --- human layer (kept separate from the RL recommendation above) ---
    human_reviewed: bool
    human_decision_action: str          # "" if never reviewed
    override_status: str                # see VALID_OVERRIDE_STATUSES
    override_reason: str

    # --- what actually gets executed ---
    final_action: str

    limitations_note: str = (
        "Synthetic simulator; heuristic (non-calibrated) confidence; "
        "prototype/demo, not validated underwriting policy."
    )

    def to_dict(self) -> dict:
        return asdict(self)


class HumanReviewGate:
    """Wraps a trained agent. Call decide() instead of agent.act() directly."""

    def __init__(self, agent: QLearningAgent, config: ReviewConfig = None,
                 audit_log_path: str = "audit_log.jsonl"):
        self.agent = agent
        self.config = config or ReviewConfig()
        self.audit_log_path = audit_log_path

    def decide(self, obs, episode_id, borrower_id, borrower_type, month,
               borrower_context: dict, recent_recommendations: list,
               human_decision_fn=None) -> tuple:
        """
        human_decision_fn: optional callable(context_dict) -> (decision, action_name_or_None, reason).
            decision is one of: "approve", "modify", "defer", "escalate".
            For "approve"/"defer"/"escalate", action_name_or_None may be None
            (the RL recommendation is used for "approve"/"defer"; "escalate"
            falls back to the conservative default "maintain" unless an
            explicit alternative is supplied).
            For "modify", action_name_or_None MUST be one of ACTIONS.

        Returns (final_action_idx, ReviewDecision).
        """
        inspection: RLInspection = inspect(self.agent, obs, self.config)
        flags = evaluate_flags(obs, inspection, borrower_context, recent_recommendations, self.config)
        flagged = len(flags) > 0

        human_reviewed = False
        human_decision_action = ""
        override_reason = ""
        final_action = inspection.recommended_action
        override_status = "approved"

        if flagged:
            if human_decision_fn is not None:
                decision, action_name, reason = human_decision_fn({
                    "episode_id": episode_id,
                    "borrower_id": borrower_id,
                    "borrower_type": borrower_type,
                    "month": month,
                    "obs": obs,
                    "inspection": inspection,
                    "flags": flags,
                })
                human_reviewed = True
                override_reason = reason or ""

                if decision == "approve":
                    final_action = inspection.recommended_action
                    override_status = "approved"
                    human_decision_action = inspection.recommended_action
                elif decision == "modify":
                    if action_name not in ACTIONS:
                        raise ValueError(f"'modify' requires a valid action, got {action_name!r}")
                    final_action = action_name
                    human_decision_action = action_name
                    override_status = "overridden" if action_name != inspection.recommended_action else "approved"
                elif decision == "defer":
                    # The environment requires SOME action every month, so
                    # "defer" cannot mean "do nothing" - it executes the RL
                    # recommendation this month while explicitly marking the
                    # decision as unresolved for follow-up, rather than
                    # silently counting it as reviewed-and-approved.
                    final_action = inspection.recommended_action
                    human_decision_action = ""
                    override_status = "deferred"
                elif decision == "escalate":
                    final_action = action_name if action_name in ACTIONS else "maintain"
                    human_decision_action = final_action
                    override_status = "escalated"
                else:
                    raise ValueError(f"Unknown human decision: {decision!r}")
            else:
                # Flagged but nobody reviewed it: keep the RL recommendation
                # executing (never silently swapped), but record honestly
                # that no human has looked at it yet.
                override_status = "needs_review"

        assert override_status in VALID_OVERRIDE_STATUSES

        record = ReviewDecision(
            timestamp=time.time(),
            episode_id=episode_id,
            borrower_id=borrower_id,
            borrower_type=borrower_type,
            month=month,
            obs=[round(float(x), 4) for x in obs],
            discretized_state=list(inspection.discretized_state),
            rl_recommended_action=inspection.recommended_action,
            q_values=inspection.q_values,
            q_margin=round(inspection.q_margin, 4),
            confidence_label=inspection.confidence_label,
            state_seen_in_training=inspection.state_seen_in_training,
            rl_explanation=inspection.explanation,
            review_flags=[f.code for f in flags],
            review_flag_details=[f.message for f in flags],
            flagged=flagged,
            human_reviewed=human_reviewed,
            human_decision_action=human_decision_action,
            override_status=override_status,
            override_reason=override_reason,
            final_action=final_action,
        )
        audit_log.append_record(self.audit_log_path, record.to_dict())

        return ACTIONS.index(final_action), record


# --- v3 note (not built) ---
# To turn overrides into training signal later: load audit_log.jsonl,
# filter for override_status == "overridden", then for each row call
# agent.update(state, ACTIONS.index(final_action), reward=+bonus, next_state,
# done=False) with a hand-picked positive bonus - nudging the Q-table toward
# the human's choice without a full retrain. Treat it as a small number of
# high-trust corrections, not a replacement for training.
