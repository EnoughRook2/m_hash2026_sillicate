# Microfinance Repayment RL - Starter Kit

## Files

- `microfinance_env.py` - the simulator + Gymnasium environment. This is the
  part that matters most; everything else is disposable.
- `baseline_policy.py` - rule-based policy. Use this as your fallback demo
  and as the benchmark to prove RL adds value.
- `q_learning_agent.py` - tabular Q-learning (no torch/deep learning needed).
  Run directly (`python3 q_learning_agent.py`) to train and save `q_table.pkl`.
- `evaluate.py` - runs both policies on fresh simulated borrowers, prints a
  comparison table, and prints readable trajectories with plain-language
  "why" flags for a few sample borrowers.

## How to run it

```
pip install gymnasium numpy   # no torch needed for the tabular version
python3 q_learning_agent.py   # trains, saves q_table.pkl, ~10 seconds
python3 evaluate.py           # compares RL vs. baseline, prints trajectories
```

## What actually happened when I built this (read this before your demo)

I trained the agent, then checked what action it was actually choosing -
**not just the headline numbers** - and found it was picking "moratorium"
85% of the time regardless of borrower state. That's not a policy, that's
reward hacking.

The cause: the original reward measured `payment_made / required`, where
`required` was itself set by the chosen action. Picking moratorium shrinks
`required` down to just interest, so the agent could clear its own lowered
bar for near-free reward every month, and the "loan repaid" terminal bonus
only checked arrears (which stayed near-zero under moratorium) rather than
whether principal actually got paid down. The agent found the shortcut
instead of the intended behavior. This is the textbook version of the
"reward feels counterintuitive" problem - the fix isn't philosophical, it's
usually a specific measurement bug like this one.

**The fix** (already applied in this code):
1. Collection reward is now measured against the *original* scheduled EMI,
   not the action-adjusted required amount - so relief has a real, visible
   cost that has to be earned back.
2. Moratorium now carries an explicit direct cost (`w5_relief_cost`).
3. The terminal "repaid" bonus now checks actual principal paid down, not
   just low arrears.

After the fix, action usage is spread across all three actions and the
default rate improved: **RL ~15-17% default vs. baseline ~23%** across 300
fresh simulated borrowers, with matching recovery rates (~90%).

**One thing I'm flagging rather than hiding**: the RL policy's *average
reward* is currently lower than the baseline's, even though its default
rate is better. That's because it restructures more often (reduce_extend/
moratorium), each of which now carries a small cost (`w3`, `w5`) - so it's
trading "some restructuring overhead" for "fewer defaults." Whether that's
the right trade-off is a judgment call for your team, and it's exactly the
kind of thing to tune in the "reward shaping" day of the plan:

- If you want RL to look more clearly superior: raise `w4_missed_payment`
  and the default penalty (currently -10.0) relative to `w3`/`w5` - this
  tells the agent that avoiding default matters even more than avoiding
  restructuring overhead.
- If you want a more conservative, judge-defensible agent: keep tuning
  slowly and log the resulting action distribution after every change
  (see the diagnostic snippet at the bottom of this file) - if one action
  dominates again, you've likely reintroduced a reward-hacking shortcut.

## Known simplifications (say these out loud in your demo, don't wait to be asked)

- Borrower "willingness to draw down savings" and "recovery after relief"
  are modeled with simple fixed rules, not learned from real behavior -
  this is the sim-to-real gap. Your RL policy is optimal *for this
  simulator's assumptions*, not proven optimal for real borrowers.
- Only 3 restructuring actions are modeled (no partial liquidation /
  write-off yet - that's Soham's "??" suggestion from your parameters doc,
  still open).
- Tabular Q-learning discretizes state into ~3-4 bins per feature. It's
  fast and transparent but coarser than a neural policy. If you have spare
  compute (e.g. Colab with GPU), swapping to stable-baselines3 PPO needs
  zero changes to `microfinance_env.py` - only the training loop changes.

## Human review layer (v2)

The first pass at this (a single `HumanReviewGate` class checking one Q-margin
threshold plus a few inline conditions, logging to `review_log.csv`) has been
replaced by a more complete layer. It's a *replacement*, not a bugfix on top -
`review_log.csv` is left exactly as it was, as a historical record of that
first version; nothing after this point writes to it.

Files:
- `review_config.py` - every threshold used by the review layer, in one
  place. Nothing is a magic number scattered through the other files.
- `rl_inspection.py` - reads Q-values, margin, and a heuristic confidence
  label directly off the trained agent for the current state. Also reports
  which risk signals are present in the raw observation, and produces an
  RL-specific explanation. It deliberately never uses `baseline_policy.py`'s
  reasons (those explain the *baseline's* decision, not the RL agent's -
  conflating them was the old evaluator's flaw), and it never claims a
  signal "caused" the recommendation - only that it's present alongside it.
- `review_flags.py` - the configurable review triggers (low confidence, high
  DTI, low buffer, recent missed payments, high arrears, low remaining
  tenure, moratorium near term end, restructuring cap, repeated moratorium,
  unusual/compounding stress combinations, serious distress paired with
  `maintain`, and unseen states).
- `audit_log.py` - append-only JSONL log. Every decision this layer makes -
  auto-executed or human-reviewed - is written to `audit_log.jsonl`, keeping
  the RL recommendation, the human's decision (if any), the final executed
  action, override status, and override reason as separate fields. The RL
  recommendation is never overwritten by a human decision.
- `human_review.py` - `HumanReviewGate.decide(...)`, which composes the
  three modules above into one call, wraps the trained agent, and appends to
  the audit log.
- `review_diagnostics.py` - action distribution (overall and by borrower
  type), action-switch frequency, low-confidence rate, unseen-state rate,
  review-trigger frequency, human override rate and reasons, and the
  moratorium-concentration warning below.
- `review.py` - the CLI: `python review.py` for an interactive terminal
  review workflow, `python review.py --auto` for a deterministic
  non-interactive demo. Prints a month-by-month decision timeline and the
  diagnostics report at the end.
- `demo_human_review.py` - kept only as a backward-compatible entry point;
  it now just calls into `review.py`'s auto-demo function rather than
  duplicating logic.
- `test_human_review.py` - unit tests for Q-value extraction, confidence
  bands, every review flag, state familiarity, override recording, audit
  logging, and diagnostics.

Re-running the same 15-borrower demo that produced v1's 46% flag rate gives
a similar overall rate (roughly half of decisions get flagged), but the
richer trigger breakdown makes *why* visible in a way v1's flat reason
string didn't: `excessive_restructuring` and `high_arrears` are still the
two biggest contributors, confirming v1's own observation that continuous
monthly re-flagging of an already-at-risk account is the main driver, not a
bug. That's a review-workload tuning decision for your team via
`review_config.py`, not something silently fixed here.

**Diagnostic warning, unchanged in spirit from before:**
```
WARNING:
Moratorium selected in 73.4% of decisions.

Configured diagnostic threshold: 70%.

Possible reward/policy failure.
Inspect policy behavior before deployment.
```
`review_diagnostics.print_diagnostics_report()` prints this automatically
whenever moratorium's share crosses `ReviewConfig.excessive_moratorium_diagnostic_pct`.
The response to that warning is always "go inspect the policy" - nothing in
this layer silently adjusts the policy to make the statistics look better.

**Important limitations, worth repeating out loud in a demo:**
- The environment is synthetic and borrower behavior is simulated (see
  "Known simplifications" above) - this layer doesn't change that.
- `confidence_label` is a heuristic from Q-value separation, not a
  calibrated probability of correctness.
- "State familiarity" is binary (seen vs. never-updated-during-training),
  because the tabular Q-learning agent's `q_table.pkl` stores no visit
  counts - there's no way to distinguish "seen once" from "seen often"
  without changing training.
- All review thresholds in `review_config.py` are configurable heuristics
  for this simulation/demo, not validated real-world underwriting policy.
- This is a prototype/demo, not production credit-decision software.

See `human_review.py`'s docstring for how human overrides could become
retraining signal later (not built yet).



```python
from microfinance_env import MicrofinanceEnv
from q_learning_agent import QLearningAgent, discretize
from collections import Counter

agent = QLearningAgent()
agent.load("q_table.pkl")
counts = Counter()
for i in range(200):
    env = MicrofinanceEnv(seed=1000 + i)
    obs, _ = env.reset()
    for _ in range(48):
        a = agent.act(discretize(obs), greedy=True)
        counts[a] += 1
        obs, r, term, trunc, info = env.step(a)
        if term or trunc:
            break
print(counts)  # if one action dominates >70%, suspect reward hacking again
```
