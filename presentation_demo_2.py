"""
Presentation / demo mode for the trained microfinance RL policy.

    python presentation_demo.py                       step through the bundled example
    python presentation_demo.py --case my_case.json    step through a custom scenario
    python presentation_demo.py --auto                 print all months without waiting
    python presentation_demo.py --review auto          also run months through HumanReviewGate,
                                                         auto-approving flagged decisions
    python presentation_demo.py --review interactive   escalate flagged decisions to a live
                                                         reviewer prompt (reuses review.py's prompt)

This script never trains, updates, or otherwise mutates the Q-table - it
only loads an already-trained q_table.pkl (via QLearningAgent.load) and
queries it read-only, through the same rl_inspection.py / human_review.py
code paths review.py and evaluate.py already use. Borrower statistics come
from a hand-authored JSON scenario (see presentation_scenario.py for the
schema/validation), not from MicrofinanceEnv's random simulator.

PRESENTATION / ILLUSTRATIVE MODE: scenarios run through this script are
manually constructed for demonstration and are not real borrower data, nor
sampled from or validated against the simulator's random generation. As
elsewhere in this project, the RL policy is optimal only under the
simulator's modeling assumptions, not proven optimal for real borrowers.
"""



import argparse
import json

from microfinance_env import ACTIONS, MicrofinanceEnv
from q_learning_agent import QLearningAgent, discretize
from review_config import ReviewConfig
from rl_inspection import inspect, FEATURE_NAMES
from human_review import HumanReviewGate
from review import ACTION_DISPLAY, STATUS_DISPLAY, auto_human_decision, interactive_human_decision
from presentation_scenario import build_case, iter_observations

DEFAULT_CASE_PATH = "presentation_case_example.json"
DEFAULT_MAX_RESTRUCTURES = MicrofinanceEnv().max_restructures  # same default the trained policy saw

BANNER = (
    "=" * 60 + "\n"
    "PRESENTATION / ILLUSTRATIVE MODE\n"
    "Manually authored scenario - not real borrower data, not sampled from\n"
    "the simulator. The policy underneath is unchanged: same q_table.pkl,\n"
    "same rl_inspection.py, same HumanReviewGate as the rest of the project.\n"
    + "=" * 60
)


def _fmt_inr(x):
    return f"\u20b9{x:,.0f}"


def print_month_block(case, month_stats, obs, discretized_state, insp, review_record):
    print("\n" + "=" * 60)
    print(f"MONTH {month_stats.month} / {len(case.months)}   -   {case.borrower_id} ({case.borrower_type})")
    print("=" * 60)

    print("\nInput statistics")
    print("-" * 40)
    print(f"  Income:              {_fmt_inr(month_stats.income)}")
    print(f"  DTI:                 {month_stats.dti:.2f}")
    print(f"  Savings buffer:      {_fmt_inr(month_stats.savings_buffer)}")
    print(f"  Arrears:             {_fmt_inr(month_stats.arrears)}")
    print(f"  Consecutive missed:  {month_stats.consecutive_missed}")

    print("\nRL observation")
    print("-" * 40)
    for name, val in zip(FEATURE_NAMES, obs):
        print(f"  {name:22s} {val:.3f}")

    print("\nDiscretized state (bin index per feature)")
    print("-" * 40)
    for name, val in zip(FEATURE_NAMES, discretized_state):
        print(f"  {name:22s} {val}")

    print("\nQ-values")
    print("-" * 40)
    for action in ACTIONS:
        print(f"  {ACTION_DISPLAY[action]:14s} {insp.q_values[action]:.2f}")
    print(f"\n  Best action:   {ACTION_DISPLAY[insp.recommended_action]}")
    print(f"  Q-margin:      {insp.q_margin:.2f}")
    print(f"  Confidence:    {insp.confidence_label}")
    if not insp.state_seen_in_training:
        print("  NOTE: this exact discretized state was never visited during training.")

    print("\nRL recommendation")
    print("-" * 40)
    print(f"  {ACTION_DISPLAY[insp.recommended_action].upper()}")

    print("\nWhy this recommendation is being surfaced")
    print("-" * 40)
    print(f"  {insp.explanation}")

    if review_record is not None:
        print("\nHuman review layer")
        print("-" * 40)
        print(f"  Flagged for review:  {'yes' if review_record.flagged else 'no'}")
        for msg in review_record.review_flag_details:
            print(f"    [!] {msg}")
        print(f"  Final action:        {ACTION_DISPLAY[review_record.final_action]}")
        print(f"  Status:              {STATUS_DISPLAY[review_record.override_status]}")
        if review_record.human_decision_action:
            print(f"  Human decision:      {ACTION_DISPLAY[review_record.human_decision_action]}")
        if review_record.override_reason:
            print(f"  Reason given:        {review_record.override_reason}")


def print_comparison_table(rows):
    print("\n" + "=" * 60)
    print("MONTH-BY-MONTH SUMMARY")
    print("=" * 60)
    header = f"{'Month':>5} {'Income':>10} {'DTI':>6} {'Action':>14} {'Q-margin':>9} {'Confidence':>11}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['month']:>5} {_fmt_inr(r['income']):>10} {r['dti']:>6.2f} "
              f"{ACTION_DISPLAY[r['action']]:>14} {r['q_margin']:>9.2f} {r['confidence']:>11}")

    print("\nAction timeline:")
    print("  " + "  ->  ".join(f"M{r['month']}:{ACTION_DISPLAY[r['action']]}" for r in rows))


def run(case, agent, config, review_mode, audit_log_path, auto):
    gate = HumanReviewGate(agent, config=config, audit_log_path=audit_log_path) if review_mode != "off" else None
    human_decision_fn = {"off": None, "auto": auto_human_decision, "interactive": interactive_human_decision}[review_mode]

    recent_recommendations = []
    restructure_count = 0
    prev_final_action = None
    rows = []

    for month_stats, obs in iter_observations(case):
        state = discretize(obs)
        insp = inspect(agent, obs, config)

        review_record = None
        if gate is not None:
            borrower_context = {"restructure_count": restructure_count, "max_restructures": DEFAULT_MAX_RESTRUCTURES}
            _, review_record = gate.decide(
                obs, episode_id=f"presentation_{case.borrower_id}", borrower_id=case.borrower_id,
                borrower_type=case.borrower_type, month=month_stats.month,
                borrower_context=borrower_context, recent_recommendations=recent_recommendations,
                human_decision_fn=human_decision_fn,
            )
            recent_recommendations.append(review_record.rl_recommended_action)
            final_action = review_record.final_action
        else:
            final_action = insp.recommended_action

        if prev_final_action is not None and final_action != prev_final_action:
            restructure_count += 1
        prev_final_action = final_action

        print_month_block(case, month_stats, obs, state, insp, review_record)
        rows.append({
            "month": month_stats.month, "income": month_stats.income, "dti": month_stats.dti,
            "action": final_action, "q_margin": insp.q_margin, "confidence": insp.confidence_label,
        })

        if not auto and month_stats.month < len(case.months):
            input("\nPress ENTER for the next month...")

    print_comparison_table(rows)
    return rows


# The three hero cases for the live demo (see HERO_CASES_README.md for the
# narrative behind each one and how the numbers were chosen). Each was
# checked against the actual trained q_table.pkl - see that file for the
# real Q-values/margins/confidence, not assumed outcomes.
HERO_CASES = [
    ("MAINTAIN", "case_maintain.json"),
    ("REDUCE/EXTEND", "case_reduce_extend.json"),
    ("MORATORIUM", "case_moratorium.json"),
]


def print_demo_summary(results):
    print("\n" + "#" * 60)
    print("DEMO SUMMARY - final recommendation per hero scenario")
    print("#" * 60)
    for label, case_id, final_action in results:
        print(f"  {case_id:32s} -> {ACTION_DISPLAY[final_action].upper()} "
              f"(intended: {label})")


def main():
    parser = argparse.ArgumentParser(description="Step through a manually-authored borrower "
                                                   "scenario against the trained RL policy.")
    parser.add_argument("--case", default=DEFAULT_CASE_PATH,
                         help="Path to a JSON presentation case. Ignored if --demo is set.")
    parser.add_argument("--demo", action="store_true",
                         help="Run all three hero scenarios (MAINTAIN, REDUCE/EXTEND, "
                              "MORATORIUM) back to back, then print a one-line summary of "
                              "each scenario's final recommendation.")
    parser.add_argument("--auto", action="store_true", help="Print all months without waiting for ENTER.")
    parser.add_argument("--review", choices=["off", "auto", "interactive"], default="off",
                         help="off: pure RL recommendation only. auto: also run through "
                              "HumanReviewGate, auto-approving flagged decisions. interactive: "
                              "escalate flagged decisions to a live reviewer prompt.")
    parser.add_argument("--q-table", default="q_table.pkl")
    parser.add_argument("--audit-log", default="presentation_audit_log.jsonl",
                         help="Only used with --review auto/interactive; kept separate from the "
                              "real audit_log.jsonl so demo runs never mix into real records.")
    args = parser.parse_args()

    agent = QLearningAgent()
    agent.load(args.q_table)
    config = ReviewConfig()

    print(BANNER)

    if args.demo:
        results = []
        for label, case_path in HERO_CASES:
            print(f"\n\n{'#' * 60}\n# SCENARIO: {label}  ({case_path})\n{'#' * 60}")
            with open(case_path) as f:
                raw_case = json.load(f)
            case = build_case(raw_case)
            rows = run(case, agent, config, args.review, args.audit_log, args.auto)
            results.append((label, case_path, rows[-1]["action"]))
        print_demo_summary(results)
    else:
        with open(args.case) as f:
            raw_case = json.load(f)
        case = build_case(raw_case)
        run(case, agent, config, args.review, args.audit_log, args.auto)


if __name__ == "__main__":
    main()
