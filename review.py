"""
Human review CLI.

    python review.py             interactive terminal review
    python review.py --auto      deterministic, non-interactive demo mode

Neither mode touches training or q_table.pkl - this only wraps an
already-trained agent's recommendations. See human_review.py for the
decision logic and review_config.py for the thresholds used below.
"""

import argparse
from collections import deque

from microfinance_env import MicrofinanceEnv, ACTIONS
from q_learning_agent import QLearningAgent
from review_config import ReviewConfig
from human_review import HumanReviewGate
import review_diagnostics

ACTION_DISPLAY = {"maintain": "Maintain", "reduce_extend": "Reduce/Extend", "moratorium": "Moratorium"}
STATUS_DISPLAY = {
    "approved": "Approved", "overridden": "Overridden", "escalated": "Escalated",
    "deferred": "Deferred", "needs_review": "Needs Review",
}


def auto_human_decision(context: dict) -> tuple:
    """Deterministic stand-in reviewer for --auto: always approves the RL
    recommendation, but the decision still passes through the same
    approve/modify/defer/escalate path and audit trail a real reviewer
    would - this mode exists to demonstrate what gets surfaced, not to
    fake human judgment."""
    return "approve", None, "auto-approved (deterministic demo mode)"


def interactive_human_decision(context: dict) -> tuple:
    inspection = context["inspection"]
    flags = context["flags"]
    obs = context["obs"]
    income_cv, dti, buffer_days_scaled, seasonal_deviation, momentum, \
        arrears_ratio, consecutive_missed, months_remaining_frac = obs

    print("\n" + "=" * 40)
    print("HUMAN REVIEW REQUIRED")
    print("=" * 40)
    print(f"\nBorrower: {context['borrower_type']}")
    print(f"Month: {context['month']}")
    print(f"\nRL RECOMMENDATION: {ACTION_DISPLAY[inspection.recommended_action].upper()}")
    print("\nQ VALUES")
    for action in ACTIONS:
        print(f"  {action:14s} {inspection.q_values[action]:.2f}")
    print(f"\nQ-MARGIN: {inspection.q_margin:.2f}")
    print(f"CONFIDENCE: {inspection.confidence_label} (heuristic)")
    if not inspection.state_seen_in_training:
        print("STATE: never visited during training")
    print("\nBORROWER STATE")
    print(f"  DTI:                  {dti:.2f}")
    print(f"  Income volatility:    {income_cv:.2f}")
    print(f"  Buffer (scaled):      {buffer_days_scaled:.2f}")
    print(f"  Seasonal deviation:   {seasonal_deviation:.2f}")
    print(f"  Momentum:             {momentum:.3f}")
    print(f"  Arrears ratio:        {arrears_ratio:.2f}")
    print(f"  Consecutive missed:   {consecutive_missed:.2f}")
    print(f"  Months remaining:     {months_remaining_frac:.2f}")
    print("\nREVIEW FLAGS")
    for f in flags:
        print(f"  [!] {f.message}")
    print("\nRL EXPLANATION")
    print(f"  {inspection.explanation}")
    print("\nDECISION")
    print("  [A] Approve RL recommendation")
    print("  [M] Modify action")
    print("  [S] Skip / defer")
    print("  [R] Escalate / reject")

    choice = input("\nReviewer action: ").strip().upper()
    if choice == "A":
        return "approve", None, input("Reason (optional): ").strip()
    if choice == "M":
        print("Choose action: 0=maintain 1=reduce_extend 2=moratorium")
        idx = int(input("Action index: ").strip())
        return "modify", ACTIONS[idx], input("Reason for override: ").strip()
    if choice == "S":
        return "defer", None, input("Reason (optional): ").strip()
    if choice == "R":
        return "escalate", None, input("Reason for escalation: ").strip()
    print("Unrecognized input, treating as defer.")
    return "defer", None, "unrecognized reviewer input"


def run_episode(env, gate, episode_id, borrower_id, borrower_type, human_decision_fn, max_steps=48):
    obs, info = env.reset()
    recent_recommendations = deque(maxlen=gate.config.repeated_moratorium_window)
    trace = []
    total_r, total_scheduled, total_paid = 0.0, 0.0, 0.0
    outcome = "truncated"

    for month in range(max_steps):
        borrower_context = {
            "restructure_count": env.borrower.restructure_count,
            "max_restructures": env.max_restructures,
        }
        action_idx, record = gate.decide(
            obs, episode_id, borrower_id, borrower_type, month,
            borrower_context, list(recent_recommendations), human_decision_fn=human_decision_fn,
        )
        recent_recommendations.append(record.rl_recommended_action)

        next_obs, reward, terminated, truncated, step_info = env.step(action_idx)
        realised = step_info["realised"]
        total_scheduled += realised["required"]
        total_paid += realised["payment_made"]
        total_r += reward
        trace.append(record)

        obs = next_obs
        if terminated or truncated:
            outcome = step_info.get("outcome", "truncated")
            break

    recovery_rate = total_paid / total_scheduled if total_scheduled > 0 else 1.0
    missed_payments = sum(1 for r in trace if "recent_missed_payments" in r.review_flags)
    restructures = env.borrower.restructure_count
    overrides = sum(1 for r in trace if r.override_status == "overridden")

    return {
        "borrower_type": borrower_type, "outcome": outcome, "total_return": total_r,
        "recovery_rate": recovery_rate, "trace": trace, "restructures": restructures,
        "missed_payments": missed_payments, "overrides": overrides,
    }


def print_episode_summary(borrower_id, result):
    print(f"\n--- {borrower_id} ({result['borrower_type']}) ---")
    for i, record in enumerate(result["trace"], start=1):
        action_label = ACTION_DISPLAY[record.final_action]
        status_label = STATUS_DISPLAY[record.override_status]
        tag = f"REVIEW -> {status_label}" if record.flagged else status_label
        print(f"Month {i:2d} -> {action_label:12s} -> {tag}")
    print(f"Borrower type:   {result['borrower_type']}")
    print(f"Restructures:    {result['restructures']}")
    print(f"Missed payments: {result['missed_payments']}")
    print(f"Human overrides: {result['overrides']}")
    print(f"Outcome:         {result['outcome']}")
    print(f"Recovery rate:   {result['recovery_rate']:.0%}")
    print(f"Total return:    {result['total_return']:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Human-in-the-loop review over the trained RL policy.")
    parser.add_argument("--auto", action="store_true", help="Deterministic, non-interactive demo mode.")
    parser.add_argument("--n-borrowers", type=int, default=10)
    parser.add_argument("--seed", type=int, default=500)
    parser.add_argument("--q-table", default="q_table.pkl")
    parser.add_argument("--audit-log", default="audit_log.jsonl")
    args = parser.parse_args()

    agent = QLearningAgent()
    agent.load(args.q_table)
    config = ReviewConfig()
    gate = HumanReviewGate(agent, config=config, audit_log_path=args.audit_log)
    human_decision_fn = auto_human_decision if args.auto else interactive_human_decision

    print("NOTE: synthetic simulator, heuristic (non-calibrated) confidence, "
          "prototype/demo - not validated underwriting policy.\n")

    all_records = []
    for i in range(args.n_borrowers):
        env = MicrofinanceEnv(seed=args.seed + i)
        borrower_id = f"borrower_{args.seed + i}"
        episode_id = f"ep_{args.seed + i}"
        obs, info = env.reset()
        result = run_episode(env, gate, episode_id, borrower_id, info["borrower_type"], human_decision_fn)
        print_episode_summary(borrower_id, result)
        all_records.extend(r.to_dict() for r in result["trace"])

    review_diagnostics.print_diagnostics_report(all_records, config)
    print(f"\nFull audit log written to {args.audit_log}")


def run_auto_demo(n_borrowers=15, seed=500, q_table="q_table.pkl", audit_log_path="audit_log.jsonl"):
    """Programmatic equivalent of `python review.py --auto`, kept for
    backward-compatible callers (see demo_human_review.py)."""
    agent = QLearningAgent()
    agent.load(q_table)
    config = ReviewConfig()
    gate = HumanReviewGate(agent, config=config, audit_log_path=audit_log_path)

    all_records = []
    for i in range(n_borrowers):
        env = MicrofinanceEnv(seed=seed + i)
        borrower_id = f"borrower_{seed + i}"
        episode_id = f"ep_{seed + i}"
        obs, info = env.reset()
        result = run_episode(env, gate, episode_id, borrower_id, info["borrower_type"], auto_human_decision)
        all_records.extend(r.to_dict() for r in result["trace"])

    flagged = sum(1 for r in all_records if r["flagged"])
    total = len(all_records)
    print(f"{flagged}/{total} decisions flagged for review ({flagged/total:.1%})" if total else "No decisions.")
    review_diagnostics.print_diagnostics_report(all_records, config)
    print(f"Full audit log written to {audit_log_path}")


if __name__ == "__main__":
    main()
