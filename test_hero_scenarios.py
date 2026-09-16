"""
Regression tests for the three hero presentation scenarios.

These load the ACTUAL trained q_table.pkl (not a mock) and assert that the
final month of each hero scenario produces the intended action with a real
Q-margin - i.e. the demo's headline claims ("this borrower gets MAINTAIN",
etc.) are checked against the real model on every run, not just eyeballed
once. If q_table.pkl is ever retrained, these tests will fail loudly if the
new table no longer supports the demo - which is the point: it tells you
to re-run the scenario-tuning pass in explore.py rather than present a
scenario the current model doesn't actually agree with.

Run with:
    python -m unittest test_hero_scenarios -v
"""

import json
import unittest

from presentation_scenario import build_case, iter_observations
from q_learning_agent import QLearningAgent
from review_config import ReviewConfig
from rl_inspection import inspect

Q_TABLE_PATH = "q_table.pkl"
MIN_ACCEPTABLE_MARGIN = 1.0  # below this, a demo recommendation is too close to call


def _load_case(path):
    with open(path) as f:
        raw = json.load(f)
    return build_case(raw)


def _final_inspection(case, agent, config):
    *_, (last_month, last_obs) = iter_observations(case)
    return last_month, inspect(agent, last_obs, config)


class TestHeroScenarios(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.agent = QLearningAgent()
        cls.agent.load(Q_TABLE_PATH)
        cls.config = ReviewConfig()

    def _assert_final_action(self, case_path, expected_action):
        case = _load_case(case_path)
        month, insp = _final_inspection(case, self.agent, self.config)
        self.assertEqual(
            insp.recommended_action, expected_action,
            f"{case_path} month {month.month}: expected {expected_action}, "
            f"got {insp.recommended_action} (Q={insp.q_values})",
        )
        self.assertGreaterEqual(
            insp.q_margin, MIN_ACCEPTABLE_MARGIN,
            f"{case_path} month {month.month}: Q-margin {insp.q_margin:.2f} is too "
            f"thin for a confident live-demo recommendation",
        )
        return insp

    def test_maintain_case_recommends_maintain(self):
        self._assert_final_action("case_maintain.json", "maintain")

    def test_reduce_extend_case_recommends_reduce_extend(self):
        self._assert_final_action("case_reduce_extend.json", "reduce_extend")

    def test_moratorium_case_recommends_moratorium(self):
        self._assert_final_action("case_moratorium.json", "moratorium")

    def test_hero_states_were_actually_seen_during_training(self):
        # A confident-looking recommendation from an UNSEEN state is really
        # just the untouched all-zero default - make sure none of the hero
        # cases are secretly relying on that.
        for path in ["case_maintain.json", "case_reduce_extend.json", "case_moratorium.json"]:
            case = _load_case(path)
            _, insp = _final_inspection(case, self.agent, self.config)
            self.assertTrue(
                insp.state_seen_in_training,
                f"{path}: final month's discretized state was never visited during "
                f"training - its Q-values are meaningless zeros, not a real recommendation.",
            )

    def test_dti_is_mechanically_consistent_with_stated_emi(self):
        # Guards against exactly the bug found during review: a scenario can
        # look plausible while its displayed DTI silently disagrees with
        # original_emi / trailing income - inviting an audience member with
        # a calculator to catch the mismatch mid-demo.
        for path in ["case_maintain.json", "case_reduce_extend.json", "case_moratorium.json"]:
            with open(path) as f:
                raw = json.load(f)
            emi = raw["original_emi"]
            hist = []
            for m in raw["months"]:
                hist.append(m["income"])
                implied_dti = emi / (sum(hist) / len(hist))
                self.assertAlmostEqual(
                    m["dti"], implied_dti, delta=0.01,
                    msg=f"{path} month {m['month']}: stated dti={m['dti']} but "
                        f"original_emi/trailing_income implies {implied_dti:.3f}",
                )

    def test_maintain_case_is_consistent_across_all_months(self):
        # Not just the final month - the whole trajectory should read as a
        # coherent "healthy borrower" story for a non-technical audience.
        case = _load_case("case_maintain.json")
        for month, obs in iter_observations(case):
            insp = inspect(self.agent, obs, self.config)
            self.assertEqual(insp.recommended_action, "maintain",
                              f"case_maintain.json month {month.month} recommended "
                              f"{insp.recommended_action}, not maintain")

    def test_reduce_extend_case_starts_as_maintain(self):
        # The story is "healthy, then pressure appears" - the early months
        # should read as maintain, not jump straight to restructuring or
        # (worse) moratorium, before the transition happens.
        case = _load_case("case_reduce_extend.json")
        results = list(iter_observations(case))
        for month, obs in results[:-1]:
            insp = inspect(self.agent, obs, self.config)
            self.assertEqual(insp.recommended_action, "maintain",
                              f"case_reduce_extend.json month {month.month} recommended "
                              f"{insp.recommended_action}, expected maintain before the transition")

    def test_moratorium_case_is_not_moratorium_before_the_shock(self):
        # The whole point of this scenario is a genuine income DIP causing
        # moratorium - it should NOT already be recommending moratorium
        # while the borrower is still healthy.
        case = _load_case("case_moratorium.json")
        results = list(iter_observations(case))
        for month, obs in results[:-1]:
            insp = inspect(self.agent, obs, self.config)
            self.assertNotEqual(insp.recommended_action, "moratorium",
                                 f"case_moratorium.json month {month.month} already "
                                 f"recommended moratorium before the income dip")


if __name__ == "__main__":
    unittest.main()
