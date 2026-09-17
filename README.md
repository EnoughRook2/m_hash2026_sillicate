# Microfinance Repayment RL - Starter Kit

A tabular Q-learning agent that recommends month-to-month repayment
structuring (maintain / reduce & extend / moratorium) for microfinance
borrowers on a synthetic simulator, wrapped in a human-in-the-loop review
layer with a CLI and a browser UI, an audit trail, and diagnostics.

## Core RL pipeline

- `microfinance_env.py` - the simulator + Gymnasium environment: a
  `BorrowerSimulator` (income trend + seasonality + shocks + noise, and how
  a borrower responds to different repayment terms) wrapped as a
  `MicrofinanceEnv` so any RL algorithm can be trained on it. This is the
  part that matters most; everything else is disposable.
- `baseline_policy.py` - rule-based policy (`rule_based_policy`). Serves as
  a fallback demo and as the benchmark that proves the RL agent adds value
  over a sensible hand-written policy.
- `q_learning_agent.py` - tabular Q-learning agent (`QLearningAgent`),
  discretizing the 8-dim continuous observation into bins so no
  torch/deep-learning dependency is needed. Run directly
  (`python q_learning_agent.py`) to train and save `q_table.pkl`.
- `q_table.pkl` - the trained Q-table produced by `q_learning_agent.py`;
  loaded read-only by every other script (`review.py`, `evaluate.py`,
  `presentation_demo_2.py`, `review_ui.py`, the test suite).
- `evaluate.py` - runs both the RL policy and the baseline on fresh
  simulated borrowers, prints a comparison table, and prints readable
  trajectories with plain-language "why" explanations for a few sample
  borrowers. This is the explainability/demo layer for the raw policy.

## RL inspection & explainability

- `rl_inspection.py` - reads directly off the trained agent's Q-table and
  the current discretized state to produce an `RLInspection` (Q-values per
  action, margin between best/second-best, a heuristic, non-calibrated
  `confidence_label`, and whether the state was ever seen in training). It
  deliberately never claims a feature "caused" a recommendation, and never
  mixes in the rule-based baseline's reasoning.

## Human-in-the-loop review layer

- `review_config.py` - `ReviewConfig`, a single frozen dataclass holding
  every threshold used by the review layer (confidence margins,
  borrower-state stress thresholds, the "unusual combination" signal
  count, etc.). Centralizing these here means changing a number once
  instead of hunting through multiple files. None of these thresholds are
  validated real-world underwriting policy.
- `review_flags.py` - `evaluate_flags()`, a set of small independent
  checks (low confidence, unseen state, high DTI, thin buffer, missed
  payments, etc.) driven entirely by `ReviewConfig`. A decision can trigger
  any number of flags at once, and all of them are kept for the audit
  trail and the reviewer.
- `human_review.py` - the `HumanReviewGate` (v2) that sits between the
  agent's recommendation and actually executing an action, and
  `run_episode()`, which steps a full borrower episode through it. It
  produces a `ReviewDecision` record that keeps
  `rl_recommended_action` (never mutated), `human_decision_action`,
  `final_action`, `override_status` (approved / overridden / escalated /
  deferred / needs_review), and `override_reason` as separate fields. This
  is "shadow mode": it never changes how the agent was trained and can be
  dropped on top of an already-trained `q_table.pkl` with zero retraining.
  It supersedes an earlier v1 gate; see `audit_log.py` for how the two
  logs relate.
- `audit_log.py` - append-only JSONL persistence (`append_record`,
  `read_records`) for every `HumanReviewGate` decision, written to
  `audit_log.jsonl`. Deliberately a new, separate log stream from the
  earlier shadow-mode layer's `review_log.csv` (older/narrower schema);
  that file is left untouched as a historical record, and only
  `audit_log.jsonl` is written to going forward.
- `audit_log.jsonl` - the accumulated audit trail written by
  `audit_log.py`, one JSON record per reviewed decision.
- `review_diagnostics.py` - pure functions over a list of audit-log record
  dicts (action distribution, action-switch frequency, low-confidence
  rate, unseen-state rate, human-override rate, moratorium percentage,
  etc.) plus `print_diagnostics_report()`, the only function that formats
  output. Nothing here modifies records or the policy - an exceeded
  threshold only produces a printed warning to go inspect the policy.

## Running the review layer

- `review.py` - the human-review CLI.
  - `python review.py` - interactive terminal review
    (`interactive_human_decision()` prints each flagged decision and
    blocks on `input()`).
  - `python review.py --auto` - deterministic, non-interactive demo mode
    (`auto_human_decision()` always approves, but still goes through the
    same approve/modify/defer/escalate path and audit trail a real
    reviewer would).
  - Neither mode touches training or `q_table.pkl`; both only wrap an
    already-trained agent's recommendations through `HumanReviewGate`.
- `review_ui.py` - a second, browser-based front end for the exact same
  `HumanReviewGate.decide()` / `review.run_episode()` workflow `review.py`
  already runs in the terminal. It does not duplicate any decision logic,
  threshold, or Q-table access - `review.py`'s
  `interactive_human_decision()` blocks on `input()`; this file's
  `web_human_decision_fn()` renders the identical information (RL
  recommendation, Q-values, Q-margin, confidence, borrower state, review
  flags, RL explanation) as an HTML page and blocks on a browser form
  submission instead. One borrower episode runs at a time on a background
  thread; a Flask app serves the review queue, completed-episode summaries,
  and the diagnostics report once a run finishes.
  ```
  pip install flask
  python review_ui.py                              # 5 borrowers, seed 500
  python review_ui.py --n-borrowers 10 --seed 42
  # then open http://127.0.0.1:5000
  ```
  This is a local, single-reviewer demo tool (binds to `127.0.0.1` by
  default, no auth), not a multi-user review queue.
- `test_review_ui.py` - tests for `review_ui.py` using Flask's test client
  (no real network/browser) and a hand-built `FakeAgent`, so they're fast
  and don't depend on a trained `q_table.pkl`. Every helper waits for the
  specific pending `threading.Event` the worker is blocked on to actually
  be consumed (identity check, not a sleep) before posting the next
  decision, to avoid the race where a real browser reviewer can't hit but
  a tight test loop can. Run with `python -m unittest test_review_ui -v`.
- `demo_human_review.py` - **deprecated** entry point kept only for
  backward compatibility; it's a thin redirect to `review.py`'s
  `run_auto_demo()`. Prefer `python review.py --auto` going forward.
- `test_human_review.py` - tests for the human-review layer
  (`rl_inspection`, `review_flags`, `HumanReviewGate`, `audit_log`,
  `review_diagnostics`) using a tiny hand-built fake agent/q_table rather
  than the real trained model, so they're fast and don't depend on
  training having run. Run with `python -m unittest test_human_review -v`.

## Presentation / demo scenarios

- `presentation_scenario.py` - converts manually-authored, human-readable
  monthly borrower statistics (income, DTI, savings buffer, arrears, etc.)
  into the exact same 8-dim RL observation `MicrofinanceEnv._observe()`
  produces during training, by building a stand-in borrower object and
  calling `_observe()` directly rather than re-deriving any feature
  formula. `build_case()` validates and parses a scenario dict;
  `iter_observations()` yields `(month, obs)` pairs. Never touches
  `q_table.pkl` or mutates the environment/agent.
- `presentation_demo_2.py` - presentation/demo mode for the trained
  policy. Loads `q_table.pkl` read-only and queries it through the same
  `rl_inspection.py` / `human_review.py` code paths `review.py` and
  `evaluate.py` use, but steps through a hand-authored JSON scenario
  instead of `MicrofinanceEnv`'s random simulator.
  ```
  python presentation_demo_2.py                       # step through the bundled example
  python presentation_demo_2.py --case my_case.json    # step through a custom scenario
  python presentation_demo_2.py --auto                 # print all months without waiting
  python presentation_demo_2.py --review auto           # also run months through HumanReviewGate, auto-approving
  python presentation_demo_2.py --review interactive    # escalate flagged decisions to a live reviewer prompt
  ```
  Scenarios run through this script are illustrative, hand-constructed
  demo data - not real borrower data, and not sampled from or validated
  against the simulator's random generation.
- `test_presentation_scenario.py` - tests for the scenario-conversion
  layer (`build_case` validation: missing borrower id, bad borrower type,
  non-sequential months, negative income, arrears without an original EMI,
  etc.). Run with `python -m unittest test_presentation_scenario -v`.
- `test_hero_scenarios.py` - regression tests for the hero presentation
  scenarios. These load the **actual** trained `q_table.pkl` (not a mock)
  and assert the final month of each scenario produces the intended action
  with a real Q-margin, so the demo's headline claims are checked against
  the live model on every run rather than eyeballed once. If `q_table.pkl`
  is ever retrained and no longer supports a scenario, this test fails
  loudly. Run with `python -m unittest test_hero_scenarios -v`.
- `case_maintain.json`, `case_moratorium.json`, `case_reduce_extend.json` -
  hand-authored demo scenarios (one per `ACTIONS` outcome) in the schema
  `presentation_scenario.py` expects, used by the hero-scenario tests and
  as ready-made `--case` inputs for `presentation_demo_2.py`.
- `presentation_case_example.json` - the default bundled example scenario
  `presentation_demo_2.py` loads when no `--case` is given.
- `testing_com.ipynb` - a scratch notebook used to explore/tune
  presentation scenarios and inspect the trained agent's behavior
  interactively.

## How to run it

```
pip install gymnasium numpy   # no torch needed for the tabular version
python q_learning_agent.py    # trains, saves q_table.pkl
python evaluate.py            # compares RL vs. baseline, prints trajectories
```

Then, optionally:

```
python review.py                  # interactive terminal review
python review.py --auto           # non-interactive demo review

pip install flask
python review_ui.py               # browser-based review UI (http://127.0.0.1:5000)
```

## Running demo cases

Edit the `.json` file for the case and run
`python presentation_demo_2.py --case case_name.json` (note some of the
files use the file name of the case, so it's recommended not to change the
name of the case files).

## Running the tests

```
python -m unittest test_human_review -v
python -m unittest test_presentation_scenario -v
python -m unittest test_hero_scenarios -v
python -m unittest test_review_ui -v
```
