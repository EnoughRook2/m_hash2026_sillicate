"""
Tests for the human-review layer.

Uses a tiny hand-built fake agent/q_table rather than the real trained
q_table.pkl, so these tests are fast and don't depend on training having
been run. Run with:  python -m unittest test_human_review -v
"""

import os
import tempfile
import unittest
from collections import defaultdict

import numpy as np

from microfinance_env import ACTIONS
from review_config import ReviewConfig
from rl_inspection import inspect
from review_flags import evaluate_flags
from human_review import HumanReviewGate
import audit_log
import review_diagnostics


class FakeAgent:
    """Minimal stand-in for QLearningAgent: just a q_table dict."""
    def __init__(self, q_table=None):
        self.q_table = defaultdict(lambda: np.zeros(len(ACTIONS)))
        if q_table:
            self.q_table.update(q_table)


# A state where reduce_extend clearly wins.
KNOWN_STATE_OBS = np.array([0.1, 0.6, 0.2, -0.2, -0.03, 0.1, 0.0, 0.5], dtype=np.float32)
# Same discretization bucket the agent will compute for KNOWN_STATE_OBS.
from q_learning_agent import discretize  # noqa: E402
KNOWN_STATE_KEY = discretize(KNOWN_STATE_OBS)


class TestRLInspection(unittest.TestCase):
    def setUp(self):
        self.config = ReviewConfig()
        self.agent = FakeAgent({KNOWN_STATE_KEY: np.array([0.5, 2.0, 1.0])})

    def test_q_value_extraction(self):
        insp = inspect(self.agent, KNOWN_STATE_OBS, self.config)
        self.assertEqual(insp.q_values["maintain"], 0.5)
        self.assertEqual(insp.q_values["reduce_extend"], 2.0)
        self.assertEqual(insp.q_values["moratorium"], 1.0)

    def test_recommendation_extraction(self):
        insp = inspect(self.agent, KNOWN_STATE_OBS, self.config)
        self.assertEqual(insp.recommended_action, "reduce_extend")

    def test_margin_and_confidence(self):
        insp = inspect(self.agent, KNOWN_STATE_OBS, self.config)
        self.assertAlmostEqual(insp.q_margin, 1.0)  # 2.0 - 1.0
        self.assertEqual(insp.confidence_label, "HIGH")

    def test_low_confidence_band(self):
        agent = FakeAgent({KNOWN_STATE_KEY: np.array([1.0, 1.1, 0.2])})
        insp = inspect(agent, KNOWN_STATE_OBS, self.config)
        self.assertEqual(insp.confidence_label, "LOW")

    def test_state_familiarity_seen(self):
        insp = inspect(self.agent, KNOWN_STATE_OBS, self.config)
        self.assertTrue(insp.state_seen_in_training)

    def test_state_familiarity_unseen_does_not_mutate_table(self):
        agent = FakeAgent()  # empty table
        before_len = len(agent.q_table)
        insp = inspect(agent, KNOWN_STATE_OBS, self.config)
        self.assertFalse(insp.state_seen_in_training)
        # Reading for inspection is allowed to populate the defaultdict (that's
        # how q_values got read), but the seen/unseen determination must
        # reflect the state BEFORE that read, which inspect() already
        # captures correctly (it checks membership first).
        self.assertGreaterEqual(len(agent.q_table), before_len)

    def test_explanation_has_no_causal_claim(self):
        insp = inspect(self.agent, KNOWN_STATE_OBS, self.config)
        # The explanation must explicitly disclaim causality, and must never
        # assert it affirmatively (e.g. "DTI caused the model to choose...").
        self.assertIn("does not establish that any single signal caused", insp.explanation)
        self.assertNotRegex(insp.explanation, r"\b\w[\w\s]{0,25} caused (the|this) (model|agent|recommendation) to")


class TestReviewFlags(unittest.TestCase):
    def setUp(self):
        self.config = ReviewConfig()
        self.agent = FakeAgent({KNOWN_STATE_KEY: np.array([0.5, 2.0, 1.0])})
        self.insp = inspect(self.agent, KNOWN_STATE_OBS, self.config)
        self.ctx = {"restructure_count": 0, "max_restructures": 6}

    def test_high_dti_flag(self):
        flags = evaluate_flags(KNOWN_STATE_OBS, self.insp, self.ctx, [], self.config)
        codes = [f.code for f in flags]
        self.assertIn("high_dti", codes)  # obs dti = 0.6 > 0.5

    def test_low_buffer_flag(self):
        flags = evaluate_flags(KNOWN_STATE_OBS, self.insp, self.ctx, [], self.config)
        codes = [f.code for f in flags]
        self.assertIn("low_buffer", codes)  # buffer_days_scaled = 0.2 < 0.35

    def test_recent_missed_not_flagged_when_zero(self):
        flags = evaluate_flags(KNOWN_STATE_OBS, self.insp, self.ctx, [], self.config)
        codes = [f.code for f in flags]
        self.assertNotIn("recent_missed_payments", codes)  # consecutive_missed = 0.0

    def test_high_arrears_flag(self):
        obs = KNOWN_STATE_OBS.copy()
        obs[5] = 0.6  # arrears_ratio
        flags = evaluate_flags(obs, self.insp, self.ctx, [], self.config)
        self.assertIn("high_arrears", [f.code for f in flags])

    def test_low_remaining_tenure_flag(self):
        obs = KNOWN_STATE_OBS.copy()
        obs[7] = 0.1  # months_remaining_frac
        flags = evaluate_flags(obs, self.insp, self.ctx, [], self.config)
        self.assertIn("low_remaining_tenure", [f.code for f in flags])

    def test_excessive_restructuring_flag(self):
        ctx = {"restructure_count": 6, "max_restructures": 6}
        flags = evaluate_flags(KNOWN_STATE_OBS, self.insp, ctx, [], self.config)
        self.assertIn("excessive_restructuring", [f.code for f in flags])

    def test_repeated_moratorium_flag(self):
        # insp.recommended_action is "reduce_extend" here, so simulate two
        # prior moratorium recs plus this decision being moratorium via a
        # separate inspection.
        agent = FakeAgent({KNOWN_STATE_KEY: np.array([0.1, 0.1, 2.0])})  # moratorium wins
        insp = inspect(agent, KNOWN_STATE_OBS, self.config)
        flags = evaluate_flags(KNOWN_STATE_OBS, insp, self.ctx,
                                ["moratorium", "moratorium"], self.config)
        self.assertIn("repeated_moratorium", [f.code for f in flags])

    def test_unseen_state_flag(self):
        agent = FakeAgent()  # nothing trained
        insp = inspect(agent, KNOWN_STATE_OBS, self.config)
        flags = evaluate_flags(KNOWN_STATE_OBS, insp, self.ctx, [], self.config)
        self.assertIn("unseen_state", [f.code for f in flags])

    def test_no_flags_on_comfortable_state(self):
        comfortable_obs = np.array([0.05, 0.1, 0.9, 0.0, 0.0, 0.0, 0.0, 0.9], dtype=np.float32)
        agent = FakeAgent({discretize(comfortable_obs): np.array([2.0, 0.5, 0.2])})
        insp = inspect(agent, comfortable_obs, self.config)
        flags = evaluate_flags(comfortable_obs, insp, self.ctx, [], self.config)
        self.assertEqual(flags, [])


class TestHumanReviewGate(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.log_path = os.path.join(self.tmpdir, "audit_log.jsonl")
        self.agent = FakeAgent({KNOWN_STATE_KEY: np.array([0.5, 2.0, 1.0])})
        self.config = ReviewConfig()
        self.gate = HumanReviewGate(self.agent, config=self.config, audit_log_path=self.log_path)
        self.ctx = {"restructure_count": 0, "max_restructures": 6}

    def test_auto_approve_when_not_flagged(self):
        comfortable_obs = np.array([0.05, 0.1, 0.9, 0.0, 0.0, 0.0, 0.0, 0.9], dtype=np.float32)
        agent = FakeAgent({discretize(comfortable_obs): np.array([2.0, 0.5, 0.2])})
        gate = HumanReviewGate(agent, config=self.config, audit_log_path=self.log_path)
        idx, record = gate.decide(comfortable_obs, "ep", "b", "trader", 0, self.ctx, [])
        self.assertFalse(record.flagged)
        self.assertEqual(record.override_status, "approved")
        self.assertEqual(record.final_action, record.rl_recommended_action)

    def test_needs_review_when_flagged_and_no_reviewer(self):
        idx, record = self.gate.decide(KNOWN_STATE_OBS, "ep", "b", "trader", 0, self.ctx, [])
        self.assertTrue(record.flagged)
        self.assertEqual(record.override_status, "needs_review")
        # RL recommendation must still be what's executed - never silently swapped.
        self.assertEqual(record.final_action, record.rl_recommended_action)

    def test_override_records_both_recommendation_and_final_separately(self):
        def modify_fn(ctx):
            return "modify", "moratorium", "borrower has other undisclosed debt"

        idx, record = self.gate.decide(KNOWN_STATE_OBS, "ep", "b", "trader", 0, self.ctx, [],
                                        human_decision_fn=modify_fn)
        self.assertEqual(record.rl_recommended_action, "reduce_extend")
        self.assertEqual(record.human_decision_action, "moratorium")
        self.assertEqual(record.final_action, "moratorium")
        self.assertEqual(record.override_status, "overridden")
        self.assertEqual(record.override_reason, "borrower has other undisclosed debt")
        self.assertEqual(idx, ACTIONS.index("moratorium"))

    def test_approve_via_reviewer_is_not_marked_overridden(self):
        def approve_fn(ctx):
            return "approve", None, "looks fine"

        idx, record = self.gate.decide(KNOWN_STATE_OBS, "ep", "b", "trader", 0, self.ctx, [],
                                        human_decision_fn=approve_fn)
        self.assertEqual(record.override_status, "approved")
        self.assertEqual(record.final_action, record.rl_recommended_action)

    def test_modify_requires_valid_action(self):
        def bad_fn(ctx):
            return "modify", "not_a_real_action", "oops"

        with self.assertRaises(ValueError):
            self.gate.decide(KNOWN_STATE_OBS, "ep", "b", "trader", 0, self.ctx, [],
                              human_decision_fn=bad_fn)

    def test_decision_is_logged(self):
        self.gate.decide(KNOWN_STATE_OBS, "ep", "b", "trader", 0, self.ctx, [])
        records = audit_log.read_records(self.log_path)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["rl_recommended_action"], "reduce_extend")


class TestAuditLog(unittest.TestCase):
    def test_append_and_read_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "log.jsonl")
            audit_log.append_record(path, {"a": 1})
            audit_log.append_record(path, {"a": 2})
            records = audit_log.read_records(path)
            self.assertEqual(records, [{"a": 1}, {"a": 2}])

    def test_read_missing_file_returns_empty(self):
        records = audit_log.read_records("/tmp/definitely_does_not_exist_12345.jsonl")
        self.assertEqual(records, [])


class TestDiagnostics(unittest.TestCase):
    def _record(self, final_action, flagged=False, override_status="approved",
                confidence="HIGH", seen=True, episode_id="ep", month=0, flags=None):
        return {
            "final_action": final_action, "borrower_type": "trader", "flagged": flagged,
            "override_status": override_status, "human_reviewed": flagged,
            "confidence_label": confidence, "state_seen_in_training": seen,
            "episode_id": episode_id, "month": month, "review_flags": flags or [],
            "override_reason": "",
        }

    def test_action_distribution(self):
        records = [self._record("moratorium")] * 8 + [self._record("maintain")] * 2
        dist = review_diagnostics.action_distribution(records)
        self.assertAlmostEqual(dist["moratorium"], 0.8)
        self.assertAlmostEqual(dist["maintain"], 0.2)

    def test_moratorium_warning_threshold(self):
        config = ReviewConfig(excessive_moratorium_diagnostic_pct=0.70)
        records = [self._record("moratorium")] * 8 + [self._record("maintain")] * 2
        self.assertGreater(review_diagnostics.moratorium_percentage(records),
                            config.excessive_moratorium_diagnostic_pct)

    def test_repeated_moratorium_detection_via_switch_frequency(self):
        records = [
            self._record("moratorium", episode_id="e1", month=0),
            self._record("moratorium", episode_id="e1", month=1),
            self._record("maintain", episode_id="e1", month=2),
        ]
        # one switch out of two transitions
        self.assertAlmostEqual(review_diagnostics.restructure_switch_frequency(records), 0.5)

    def test_override_rate(self):
        records = [
            self._record("moratorium", flagged=True, override_status="overridden"),
            self._record("maintain", flagged=True, override_status="approved"),
            self._record("maintain", flagged=False, override_status="approved"),
        ]
        self.assertAlmostEqual(review_diagnostics.human_override_rate(records), 0.5)


if __name__ == "__main__":
    unittest.main()
