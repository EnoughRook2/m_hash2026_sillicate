"""
Browser-based UI for the human-in-the-loop review layer.

This is a second front end for the exact same HumanReviewGate.decide() /
review.run_episode() workflow review.py already runs in the terminal - it
does not duplicate any decision logic, threshold, or Q-table access.
review.py's interactive_human_decision() prints to the terminal and blocks
on input(); this file's web_human_decision_fn() renders the identical
information as an HTML page and blocks on a browser form submission
instead. Everything in between - HumanReviewGate, rl_inspection,
review_flags, review_diagnostics, run_episode - is imported unchanged.

Run:
    pip install flask
    python review_ui.py                              # 5 borrowers, seed 500
    python review_ui.py --n-borrowers 10 --seed 42
    then open http://127.0.0.1:5000

Design: one borrower episode runs at a time on a background thread. When a
decision is flagged, that thread blocks on a threading.Event until the
browser POSTs a decision to /decide, then resumes - mirroring exactly how
interactive_human_decision() blocks on input() today. This is a local,
single-reviewer demo tool (binds to 127.0.0.1 by default, no auth), not a
multi-user review queue.
"""

import argparse
import threading
from dataclasses import dataclass, field

from flask import Flask, request, redirect, url_for, render_template_string

from microfinance_env import MicrofinanceEnv, ACTIONS
from q_learning_agent import QLearningAgent
from review_config import ReviewConfig
from human_review import HumanReviewGate
from review import ACTION_DISPLAY, STATUS_DISPLAY, run_episode
import review_diagnostics
import audit_log as audit_log_module

# Same order as rl_inspection.FEATURE_NAMES / the obs array itself.
FEATURE_LABELS = [
    ("Income volatility", "{:.2f}"),
    ("DTI", "{:.2f}"),
    ("Buffer (scaled)", "{:.2f}"),
    ("Seasonal deviation", "{:.2f}"),
    ("Momentum", "{:.3f}"),
    ("Arrears ratio", "{:.2f}"),
    ("Consecutive missed", "{:.2f}"),
    ("Months remaining", "{:.2f}"),
]

app = Flask(__name__)
# Sane defaults so the app is usable (e.g. under a test client) without
# going through main()'s argument parsing; main() overrides these with the
# real CLI values before serving traffic.
app.config.setdefault("N_BORROWERS", 5)
app.config.setdefault("SEED", 500)


@dataclass
class ReviewState:
    """All mutable state the background worker and the Flask routes share.
    One instance lives for the process's lifetime - this tool assumes one
    reviewer, one run at a time, matching review.py's own single-process
    scope. _reset_state() below replaces it wholesale for a fresh run."""
    lock: threading.Lock = field(default_factory=threading.Lock)
    started: bool = False
    finished: bool = False
    pending_context: dict = None
    pending_event: threading.Event = None
    pending_response: tuple = None
    pending_error: str = ""
    completed_episodes: list = field(default_factory=list)   # [(borrower_id, result_dict), ...]
    all_records: list = field(default_factory=list)          # flat list of record dicts
    current_borrower_id: str = ""


STATE = ReviewState()


def _reset_state():
    global STATE
    STATE = ReviewState()


def web_human_decision_fn(context: dict) -> tuple:
    """Blocks the calling (background worker) thread until the browser
    submits a decision via POST /decide. Same contract as
    review.interactive_human_decision(): returns (decision, action_name_or_None, reason)."""
    event = threading.Event()
    with STATE.lock:
        STATE.pending_context = context
        STATE.pending_event = event
        STATE.pending_response = None
        STATE.pending_error = ""
    event.wait()
    with STATE.lock:
        response = STATE.pending_response
        STATE.pending_context = None
        STATE.pending_event = None
    return response


def _worker(agent, config, n_borrowers, seed, audit_log_path):
    gate = HumanReviewGate(agent, config=config, audit_log_path=audit_log_path)

    for i in range(n_borrowers):
        env = MicrofinanceEnv(seed=seed + i)
        borrower_id = f"borrower_{seed + i}"
        episode_id = f"ep_{seed + i}"
        obs, info = env.reset()
        with STATE.lock:
            STATE.current_borrower_id = borrower_id
        result = run_episode(env, gate, episode_id, borrower_id, info["borrower_type"],
                              web_human_decision_fn)
        with STATE.lock:
            STATE.completed_episodes.append((borrower_id, result))
            STATE.all_records.extend(r.to_dict() for r in result["trace"])

    with STATE.lock:
        STATE.finished = True
        STATE.current_borrower_id = ""


def _feature_rows(obs):
    return [(label, fmt.format(obs[i])) for i, (label, fmt) in enumerate(FEATURE_LABELS)]


def _diagnostics_from(all_records, config):
    if not all_records:
        return None
    moratorium_pct = review_diagnostics.moratorium_percentage(all_records)
    threshold = config.excessive_moratorium_diagnostic_pct
    return {
        "action_dist": review_diagnostics.action_distribution(all_records),
        "switch_freq": review_diagnostics.restructure_switch_frequency(all_records),
        "low_conf": review_diagnostics.low_confidence_rate(all_records),
        "unseen": review_diagnostics.unseen_state_rate(all_records),
        "override_rate": review_diagnostics.human_override_rate(all_records),
        "moratorium_pct": moratorium_pct,
        "threshold": threshold,
        "moratorium_warning": moratorium_pct > threshold,
    }


PAGE = """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Microfinance RL - Human Review</title>
{% if started and not pending and not finished %}<meta http-equiv="refresh" content="2">{% endif %}
<style>
  body { font-family: -apple-system, Segoe UI, Arial, sans-serif; max-width: 780px; margin: 2rem auto; color: #222; padding: 0 1rem; }
  h1 { font-size: 1.3rem; } h2 { font-size: 1.1rem; } h3 { font-size: 0.95rem; margin-bottom: 0.3rem; }
  .card { border: 1px solid #ddd; border-radius: 8px; padding: 1.25rem; margin-bottom: 1.25rem; }
  .flag { color: #a33; margin: 0.15rem 0; }
  table { border-collapse: collapse; width: 100%; margin-bottom: 0.75rem; }
  td, th { padding: 0.3rem 0.6rem; text-align: left; border-bottom: 1px solid #eee; }
  .qval { font-family: monospace; }
  .btn { padding: 0.5rem 1rem; margin-right: 0.5rem; border-radius: 6px; border: 1px solid #999;
         background: #f7f7f7; cursor: pointer; font-size: 0.95rem; }
  .btn:hover { background: #eee; }
  .note { color: #666; font-size: 0.85rem; }
  .error { color: #a33; font-weight: bold; }
  .status-approved { color: #2a7; } .status-overridden { color: #a73; }
  .status-needs_review, .status-escalated { color: #a33; } .status-deferred { color: #888; }
  form label { display: block; margin: 0.3rem 0; }
</style>
</head>
<body>
<h1>Microfinance RL - Human Review Queue</h1>
<p class="note">Synthetic simulator, heuristic (non-calibrated) confidence, prototype/demo - not validated underwriting policy.</p>

{% if not started %}
  <form method="post" action="{{ url_for('start') }}">
    <button class="btn" type="submit">Start review run ({{ n_borrowers }} borrowers, seed {{ seed }})</button>
  </form>

{% elif pending %}
  <div class="card">
    <h2>Review required &mdash; {{ pending.borrower_type }}, month {{ pending.month + 1 }} ({{ current_borrower_id }})</h2>

    {% if error %}<p class="error">{{ error }}</p>{% endif %}

    <p><b>RL recommendation:</b> {{ action_display[pending.inspection.recommended_action] }}</p>

    <table>
      <tr><th>Action</th><th>Q-value</th></tr>
      {% for a in actions %}
      <tr><td>{{ action_display[a] }}</td><td class="qval">{{ "%.2f"|format(pending.inspection.q_values[a]) }}</td></tr>
      {% endfor %}
    </table>
    <p>Q-margin: {{ "%.2f"|format(pending.inspection.q_margin) }}
       &nbsp;&nbsp; Confidence: {{ pending.inspection.confidence_label }}
       {% if not pending.inspection.state_seen_in_training %}<br><b>This state was never visited during training.</b>{% endif %}
    </p>

    <h3>Borrower state</h3>
    <table>
      {% for label, val in feature_rows %}
      <tr><td>{{ label }}</td><td>{{ val }}</td></tr>
      {% endfor %}
    </table>

    <h3>Review flags</h3>
    {% for f in pending.flags %}<div class="flag">[!] {{ f.message }}</div>{% else %}<p class="note">(none listed)</p>{% endfor %}

    <h3>RL explanation</h3>
    <p>{{ pending.inspection.explanation }}</p>

    <form method="post" action="{{ url_for('decide') }}">
      <label><input type="radio" name="decision" value="approve" checked> Approve RL recommendation</label>
      <label><input type="radio" name="decision" value="modify"> Modify to:
        <select name="action">
          {% for a in actions %}<option value="{{ a }}">{{ action_display[a] }}</option>{% endfor %}
        </select>
      </label>
      <label><input type="radio" name="decision" value="defer"> Defer (RL recommendation executes; flagged for follow-up)</label>
      <label><input type="radio" name="decision" value="escalate"> Escalate (uses the dropdown above, or Maintain if unset)</label>
      <p>Reason: <input type="text" name="reason" style="width: 70%;"></p>
      <button class="btn" type="submit">Submit decision</button>
    </form>
  </div>

{% elif not finished %}
  <div class="card">
    <p>Processing {{ current_borrower_id }}&hellip; this page refreshes automatically.</p>
  </div>

{% else %}
  <div class="card">
    <p><b>Review run complete.</b> {{ completed_episodes|length }} borrower(s) processed.</p>
    <form method="post" action="{{ url_for('reset') }}"><button class="btn" type="submit">Start a new run</button></form>
  </div>
{% endif %}

{% if completed_episodes %}
  <h2>Completed borrowers</h2>
  {% for borrower_id, result in completed_episodes %}
  <div class="card">
    <b>{{ borrower_id }}</b> ({{ result.borrower_type }}) &mdash; outcome: {{ result.outcome }},
    recovery rate: {{ "%.0f"|format(result.recovery_rate * 100) }}%, overrides: {{ result.overrides }}
    <table>
      <tr><th>Month</th><th>Action</th><th>Status</th></tr>
      {% for r in result.trace %}
      <tr>
        <td>{{ loop.index }}</td>
        <td>{{ action_display[r.final_action] }}</td>
        <td class="status-{{ r.override_status }}">{{ status_display[r.override_status] }}{% if r.flagged %} (flagged){% endif %}</td>
      </tr>
      {% endfor %}
    </table>
  </div>
  {% endfor %}
{% endif %}

{% if diagnostics %}
  <h2>Diagnostics</h2>
  <div class="card">
    <table>
      <tr><th>Action</th><th>Share</th></tr>
      {% for action, frac in diagnostics.action_dist.items() %}
      <tr><td>{{ action_display[action] }}</td><td>{{ "%.1f"|format(frac * 100) }}%</td></tr>
      {% endfor %}
    </table>
    <p>Action-switch frequency: {{ "%.1f"|format(diagnostics.switch_freq * 100) }}%</p>
    <p>Low-confidence decisions: {{ "%.1f"|format(diagnostics.low_conf * 100) }}%</p>
    <p>Unseen-state decisions: {{ "%.1f"|format(diagnostics.unseen * 100) }}%</p>
    <p>Human override rate: {{ "%.1f"|format(diagnostics.override_rate * 100) }}%</p>
    {% if diagnostics.moratorium_warning %}
    <p class="error">WARNING: moratorium selected in {{ "%.1f"|format(diagnostics.moratorium_pct * 100) }}%
       of decisions (threshold {{ "%.0f"|format(diagnostics.threshold * 100) }}%).
       Inspect policy behavior before deployment.</p>
    {% endif %}
  </div>
{% endif %}

</body>
</html>
"""


@app.route("/", methods=["GET"])
def index():
    with STATE.lock:
        started, finished = STATE.started, STATE.finished
        pending, error = STATE.pending_context, STATE.pending_error
        completed_episodes = list(STATE.completed_episodes)
        current_borrower_id = STATE.current_borrower_id
        all_records = list(STATE.all_records)

    feature_rows = _feature_rows(pending["obs"]) if pending else []
    diagnostics = _diagnostics_from(all_records, app.config["REVIEW_CONFIG"]) if finished else None

    return render_template_string(
        PAGE, started=started, finished=finished, pending=pending, error=error,
        completed_episodes=completed_episodes, current_borrower_id=current_borrower_id,
        actions=ACTIONS, action_display=ACTION_DISPLAY, status_display=STATUS_DISPLAY,
        feature_rows=feature_rows, diagnostics=diagnostics,
        n_borrowers=app.config["N_BORROWERS"], seed=app.config["SEED"],
    )


@app.route("/start", methods=["POST"])
def start():
    with STATE.lock:
        if not STATE.started:
            STATE.started = True
            threading.Thread(
                target=_worker,
                args=(app.config["AGENT"], app.config["REVIEW_CONFIG"],
                      app.config["N_BORROWERS"], app.config["SEED"], app.config["AUDIT_LOG"]),
                daemon=True,
            ).start()
    return redirect(url_for("index"))


@app.route("/decide", methods=["POST"])
def decide():
    decision = request.form.get("decision")
    action_name = request.form.get("action") or None
    reason = request.form.get("reason", "").strip()

    with STATE.lock:
        event = STATE.pending_event
    if event is None:
        return redirect(url_for("index"))  # nothing pending (e.g. a stale/duplicate submit); ignore

    if decision == "modify" and action_name not in ACTIONS:
        with STATE.lock:
            STATE.pending_error = f"Choose a valid action to modify to (got {action_name!r})."
        return redirect(url_for("index"))

    response = (decision, action_name if decision in ("modify", "escalate") else None, reason)
    with STATE.lock:
        STATE.pending_response = response
        STATE.pending_error = ""
    event.set()
    return redirect(url_for("index"))


@app.route("/reset", methods=["POST"])
def reset():
    with STATE.lock:
        finished = STATE.finished
    if finished:
        _reset_state()
    return redirect(url_for("index"))


def main():
    parser = argparse.ArgumentParser(description="Browser UI for the human-in-the-loop review layer.")
    parser.add_argument("--n-borrowers", type=int, default=5)
    parser.add_argument("--seed", type=int, default=500)
    parser.add_argument("--q-table", default="q_table.pkl")
    parser.add_argument("--audit-log", default="audit_log.jsonl")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    agent = QLearningAgent()
    agent.load(args.q_table)

    app.config["AGENT"] = agent
    app.config["REVIEW_CONFIG"] = ReviewConfig()
    app.config["N_BORROWERS"] = args.n_borrowers
    app.config["SEED"] = args.seed
    app.config["AUDIT_LOG"] = args.audit_log

    print("NOTE: synthetic simulator, heuristic (non-calibrated) confidence, "
          "prototype/demo - not validated underwriting policy.\n")
    print(f"Loaded {args.q_table}. Open http://127.0.0.1:{args.port} in a browser.")
    app.run(port=args.port, threaded=True)


if __name__ == "__main__":
    main()
