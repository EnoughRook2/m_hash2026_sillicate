"""
Tests for the browser review UI. Uses Flask's test client (no real network,
no real browser) plus a hand-built FakeAgent, so these are fast and don't
depend on a trained q_table.pkl. Run with:
    python -m unittest test_review_ui -v

Synchronization note: a real browser reviewer can only ever have one
decision in flight (they wait for the page to reload before submitting the
next one), but a test posting decisions in a tight loop can easily post a
second decision before the background worker thread has consumed the
first - silently overwriting it. Every helper below waits for the specific
pending Event the worker is blocked on to actually be consumed (identity
check, not just "is something pending") before moving on, rather than
guessing with a sleep.
"""

import os
import tempfile
import threading
import time
import unittest
from collections import defaultdict

import numpy as np

from microfinance_env import ACTIONS
from review_config import ReviewConfig
import audit_log
import review_ui


class FakeAgent:
    """Empty q_table => every state is 'unseen' and every decision ties at
    Q=0.0 for all actions => always flagged (low_confidence, unseen_state)
    and always recommends 'maintain' (dict-order tiebreak)."""
    def __init__(self):
        self.q_table = defaultdict(lambda: np.zeros(len(ACTIONS)))


def _wait_until(predicate, timeout=5.0, interval=0.02):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


class TestReviewUI(unittest.TestCase):
    def setUp(self):
        review_ui._reset_state()
        self.client = review_ui.app.test_client()
        self.tmpdir = tempfile.mkdtemp()
        self.audit_log_path = os.path.join(self.tmpdir, "audit_log.jsonl")
        self.agent = FakeAgent()
        self.config = ReviewConfig()
        self.worker_thread = None

    def tearDown(self):
        # Drain any still-pending decision so the worker thread finishes
        # cleanly before the next test resets the module-global STATE -
        # otherwise a leaked thread from this test could mutate the next
        # test's fresh state.
        self._approve_until_finished(max_iterations=100)
        if self.worker_thread is not None:
            self.worker_thread.join(timeout=10)
            self.assertFalse(self.worker_thread.is_alive(), "worker thread leaked into next test")

    def _start_worker(self, n_borrowers=1, seed=1):
        review_ui.STATE.started = True
        self.worker_thread = threading.Thread(
            target=review_ui._worker,
            args=(self.agent, self.config, n_borrowers, seed, self.audit_log_path),
            daemon=True,
        )
        self.worker_thread.start()

    def _submit_decision(self, **form):
        """Post one decision for whatever is currently pending, and don't
        return until the worker has actually consumed it (its pending_event
        identity changes, or the run finishes) - eliminating the race where
        a second post lands before the worker reads the first."""
        event_before = review_ui.STATE.pending_event
        resp = self.client.post("/decide", data=form)
        consumed = _wait_until(lambda: review_ui.STATE.pending_event is not event_before
                                or review_ui.STATE.finished)
        return resp, consumed

    def _approve_until_finished(self, max_iterations=100):
        if self.worker_thread is None:
            return True
        for _ in range(max_iterations):
            if review_ui.STATE.finished:
                return True
            if not _wait_until(lambda: review_ui.STATE.pending_context is not None or review_ui.STATE.finished):
                return False
            if review_ui.STATE.finished:
                return True
            _, consumed = self._submit_decision(decision="approve", reason="auto")
            if not consumed:
                return False
        return review_ui.STATE.finished

    def test_index_before_start_shows_start_button(self):
        resp = self.client.get("/")
        self.assertIn(b"Start review run", resp.data)

    def test_flagged_decision_blocks_and_page_shows_review_form(self):
        self._start_worker()
        self.assertTrue(_wait_until(lambda: review_ui.STATE.pending_context is not None),
                         "worker never reached a pending decision")
        resp = self.client.get("/")
        self.assertIn(b"Review required", resp.data)
        self.assertIn(b"Q-value", resp.data)
        self.assertIn(b"never visited during training", resp.data)  # empty FakeAgent => always unseen

    def test_approve_decision_unblocks_worker_and_completes(self):
        self._start_worker()
        self.assertTrue(_wait_until(lambda: review_ui.STATE.pending_context is not None))
        finished = self._approve_until_finished()
        self.assertTrue(finished, "worker never finished after approving")
        self.assertEqual(len(review_ui.STATE.completed_episodes), 1)

    def test_invalid_modify_action_is_rejected_without_crashing_worker(self):
        self._start_worker()
        self.assertTrue(_wait_until(lambda: review_ui.STATE.pending_context is not None))
        event_before = review_ui.STATE.pending_event

        resp = self.client.post("/decide", data={"decision": "modify", "action": "not_a_real_action", "reason": "oops"})
        self.assertEqual(resp.status_code, 302)

        resp = self.client.get("/")
        self.assertIn(b"Choose a valid action", resp.data)
        # Rejected input must NOT unblock the worker - same pending decision, not crashed.
        self.assertIs(review_ui.STATE.pending_event, event_before)
        self.assertIsNotNone(review_ui.STATE.pending_context)

    def test_modify_records_override_in_audit_log(self):
        self._start_worker()
        self.assertTrue(_wait_until(lambda: review_ui.STATE.pending_context is not None))
        _, consumed = self._submit_decision(decision="modify", action="moratorium", reason="extra risk info")
        self.assertTrue(consumed)
        self._approve_until_finished()  # drain the rest of the episode

        records = audit_log.read_records(self.audit_log_path)
        self.assertEqual(records[0]["override_status"], "overridden")
        self.assertEqual(records[0]["final_action"], "moratorium")
        self.assertEqual(records[0]["rl_recommended_action"], "maintain")
        self.assertEqual(records[0]["override_reason"], "extra risk info")

    def test_defer_keeps_rl_recommendation_as_final_action(self):
        self._start_worker()
        self.assertTrue(_wait_until(lambda: review_ui.STATE.pending_context is not None))
        _, consumed = self._submit_decision(decision="defer", action="moratorium", reason="need more info")
        self.assertTrue(consumed)
        self._approve_until_finished()

        records = audit_log.read_records(self.audit_log_path)
        self.assertEqual(records[0]["override_status"], "deferred")
        # defer must execute the RL recommendation, never the dropdown value.
        self.assertEqual(records[0]["final_action"], records[0]["rl_recommended_action"])

    def test_reset_only_works_after_finished(self):
        self._start_worker()
        self.assertTrue(_wait_until(lambda: review_ui.STATE.pending_context is not None))
        resp = self.client.post("/reset")
        self.assertEqual(resp.status_code, 302)
        # Not finished yet - reset must be a no-op, worker still mid-run.
        self.assertTrue(review_ui.STATE.started)
        self.assertIsNotNone(review_ui.STATE.pending_context)


if __name__ == "__main__":
    unittest.main()
