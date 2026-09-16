"""
Diagnostics over a HumanReviewGate audit log.

Every function here takes a list of record dicts (as produced by
ReviewDecision.to_dict() / audit_log.read_records()) and returns plain
numbers or Counters - no printing, no side effects, so each is independently
testable. print_diagnostics_report() is the only function that formats
output, and it never modifies the records or the policy: if a threshold is
exceeded, the response is a printed warning to go inspect the policy, not
an automatic change to it.
"""

from collections import Counter


def action_distribution(records: list) -> dict:
    total = len(records)
    if total == 0:
        return {}
    counts = Counter(r["final_action"] for r in records)
    return {action: count / total for action, count in counts.items()}


def action_distribution_by_borrower_type(records: list) -> dict:
    by_type = {}
    for r in records:
        by_type.setdefault(r["borrower_type"], []).append(r)
    return {btype: action_distribution(rows) for btype, rows in by_type.items()}


def moratorium_percentage(records: list) -> float:
    return action_distribution(records).get("moratorium", 0.0)


def restructure_switch_frequency(records: list) -> float:
    """Fraction of consecutive-month decisions (per episode) where the
    final action differs from the previous month's final action."""
    by_episode = {}
    for r in records:
        by_episode.setdefault(r["episode_id"], []).append(r)

    switches, transitions = 0, 0
    for rows in by_episode.values():
        rows = sorted(rows, key=lambda r: r["month"])
        for prev, cur in zip(rows, rows[1:]):
            transitions += 1
            if prev["final_action"] != cur["final_action"]:
                switches += 1
    return switches / transitions if transitions else 0.0

def low_confidence_rate(records: list) -> float:
    total = len(records)
    if total == 0:
        return 0.0
    return sum(1 for r in records if r["confidence_label"] == "LOW") / total


def unseen_state_rate(records: list) -> float:
    total = len(records)
    if total == 0:
        return 0.0
    return sum(1 for r in records if not r["state_seen_in_training"]) / total


def review_trigger_frequency(records: list) -> Counter:
    counts = Counter()
    for r in records:
        counts.update(r["review_flags"])
    return counts


def human_override_rate(records: list) -> float:
    reviewed = [r for r in records if r["human_reviewed"]]
    if not reviewed:
        return 0.0
    overridden = sum(1 for r in reviewed if r["override_status"] == "overridden")
    return overridden / len(reviewed)


def override_reasons(records: list) -> Counter:
    return Counter(r["override_reason"] for r in records
                   if r["override_status"] == "overridden" and r["override_reason"])


def print_diagnostics_report(records: list, config) -> None:
    total = len(records)
    print("=" * 60)
    print("REVIEW LAYER DIAGNOSTICS")
    print("=" * 60)
    print(f"Total decisions: {total}")
    if total == 0:
        print("No records to summarize.")
        return

    print("\nAction distribution (final action executed):")
    for action, frac in sorted(action_distribution(records).items()):
        print(f"  {action:14s} {frac:6.1%}")

    print("\nAction distribution by borrower type:")
    for btype, dist in sorted(action_distribution_by_borrower_type(records).items()):
        parts = ", ".join(f"{a}={f:.0%}" for a, f in sorted(dist.items()))
        print(f"  {btype:14s} {parts}")

    print(f"\nAction-switch frequency (month-to-month): {restructure_switch_frequency(records):.1%}")
    print(f"Low-confidence decisions:                  {low_confidence_rate(records):.1%}")
    print(f"Unseen-state decisions:                     {unseen_state_rate(records):.1%}")

    print("\nReview-trigger frequency:")
    for code, count in review_trigger_frequency(records).most_common():
        print(f"  {code:30s} {count}")

    reviewed_count = sum(1 for r in records if r["human_reviewed"])
    print(f"\nDecisions reviewed by a human: {reviewed_count}/{total}")
    print(f"Human override rate (of reviewed): {human_override_rate(records):.1%}")
    reasons = override_reasons(records)
    if reasons:
        print("Override reasons given:")
        for reason, count in reasons.most_common():
            print(f"  ({count}x) {reason}")

    moratorium_pct = moratorium_percentage(records)
    print(f"\nMoratorium selected in {moratorium_pct:.1%} of decisions.")
    print(f"Configured diagnostic threshold: {config.excessive_moratorium_diagnostic_pct:.0%}.")
    if moratorium_pct > config.excessive_moratorium_diagnostic_pct:
        print("\nWARNING:")
        print(f"Moratorium selected in {moratorium_pct:.1%} of decisions.")
        print(f"Configured diagnostic threshold: {config.excessive_moratorium_diagnostic_pct:.0%}.")
        print("Possible reward/policy failure.")
        print("Inspect policy behavior before deployment.")
    print("=" * 60)
