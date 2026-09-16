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
## Running demo cases

Edit the .json file for the cases and run presentation_demo_2.py (note some of the files use the file name of the case, so its recommended to not change the name of the case files)
