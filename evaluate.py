"""
Evaluate the trained Q-learning agent against the rule-based baseline on a
fresh batch of simulated borrowers, and print a few full trajectories with
plain-language explanations - this is your explainability layer and your
strongest demo material.
"""

import numpy as np

from microfinance_env import MicrofinanceEnv, ACTIONS
from q_learning_agent import QLearningAgent, discretize
from baseline_policy import rule_based_policy


FEATURE_NAMES = ["income_cv", "dti", "buffer_days_scaled", "seasonal_deviation",
                  "momentum", "arrears_ratio", "consecutive_missed", "months_remaining_frac"]


def run_episode(env, policy_fn, max_steps=48):
    obs, info = env.reset()
    total_r = 0.0
    total_scheduled = 0.0
    total_paid = 0.0
    outcome = "truncated"
    trace = []

    for t in range(max_steps):
        action = policy_fn(obs)
        next_obs, reward, terminated, truncated, step_info = env.step(action)
        realised = step_info["realised"]
        total_scheduled += realised["required"]
        total_paid += realised["payment_made"]
        total_r += reward
        trace.append({
            "t": t, "obs": obs, "action": action,
            "required": realised["required"], "paid": realised["payment_made"],
        })
        obs = next_obs
        if terminated or truncated:
            outcome = step_info.get("outcome", "truncated")
            break

    recovery_rate = total_paid / total_scheduled if total_scheduled > 0 else 1.0
    return {
        "outcome": outcome, "total_return": total_r,
        "recovery_rate": recovery_rate, "trace": trace,
        "borrower_type": info["borrower_type"],
    }


def compare_policies(n_episodes=300, q_table_path="q_table.pkl", seed=123):
    agent = QLearningAgent()
    agent.load(q_table_path)

    def rl_policy(obs):
        return agent.act(discretize(obs), greedy=True)

    def base_policy(obs):
        return rule_based_policy(obs)[0]

    results = {"rl": [], "baseline": []}
    for i in range(n_episodes):
        env = MicrofinanceEnv(seed=seed + i)
        results["rl"].append(run_episode(env, rl_policy))
        env2 = MicrofinanceEnv(seed=seed + i)  # same borrower, fresh env instance
        results["baseline"].append(run_episode(env2, base_policy))

    print(f"{'':12s} {'default rate':>14s} {'avg recovery':>14s} {'avg return':>12s}")
    for name in ["rl", "baseline"]:
        rows = results[name]
        default_rate = sum(r["outcome"] == "default" for r in rows) / len(rows)
        avg_recovery = np.mean([r["recovery_rate"] for r in rows])
        avg_return = np.mean([r["total_return"] for r in rows])
        print(f"{name:12s} {default_rate:13.1%} {avg_recovery:13.1%} {avg_return:11.2f}")

    return results


def explain_trajectory(result, max_rows=12):
    """Print a human-readable trace of one episode: state, action, and why."""
    print(f"\nBorrower type: {result['borrower_type']}  |  outcome: {result['outcome']}  "
          f"|  recovery rate: {result['recovery_rate']:.0%}")
    for row in result["trace"][:max_rows]:
        obs = row["obs"]
        action_name = ACTIONS[row["action"]]
        _, reasons = rule_based_policy(obs)
        flags = []
        if obs[3] < -0.15:
            flags.append("below seasonal expectation")
        if obs[2] < 0.35:
            flags.append("thin buffer")
        if obs[4] < -0.02:
            flags.append("declining trend")
        if obs[6] > 0.2:
            flags.append("recent missed payment(s)")
        flag_str = ", ".join(flags) if flags else "no stress signals"
        print(f"  month {row['t']:2d}: action={action_name:14s} "
              f"paid {row['paid']:7.0f}/{row['required']:7.0f}  [{flag_str}]")


if __name__ == "__main__":
    results = compare_policies(n_episodes=300)

    print("\n--- Sample trajectories (RL policy) ---")
    # Show one repaid and one default example, for contrast.
    rl_rows = results["rl"]
    repaid_example = next((r for r in rl_rows if r["outcome"] == "repaid"), None)
    default_example = next((r for r in rl_rows if r["outcome"] == "default"), None)
    if repaid_example:
        explain_trajectory(repaid_example)
    if default_example:
        explain_trajectory(default_example)
