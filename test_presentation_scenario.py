"""
Tests for the presentation-scenario conversion layer. Run with:
    python -m unittest test_presentation_scenario -v
"""

import unittest

from presentation_scenario import build_case, iter_observations
from rl_inspection import FEATURE_NAMES


def _base_case(months):
    return {"borrower_id": "b1", "borrower_type": "trader", "months": months}


class TestBuildCaseValidation(unittest.TestCase):
    def test_missing_borrower_id_raises(self):
        with self.assertRaises(ValueError):
            build_case({"months": [{"month": 1, "income": 10000, "dti": 0.3}]})

    def test_bad_borrower_type_raises(self):
        case = _base_case([{"month": 1, "income": 10000, "dti": 0.3}])
        case["borrower_type"] = "loan_shark"
        with self.assertRaises(ValueError):
            build_case(case)

    def test_non_sequential_months_raises(self):
        with self.assertRaises(ValueError):
            build_case(_base_case([
                {"month": 1, "income": 10000, "dti": 0.3},
                {"month": 3, "income": 9000, "dti": 0.4},
            ]))

    def test_arrears_without_original_emi_raises(self):
        with self.assertRaises(ValueError):
            build_case(_base_case([{"month": 1, "income": 10000, "dti": 0.3, "arrears": 500}]))

    def test_negative_income_raises(self):
        with self.assertRaises(ValueError):
            build_case(_base_case([{"month": 1, "income": -100, "dti": 0.3}]))

    def test_missing_income_raises(self):
        with self.assertRaises(ValueError):
            build_case(_base_case([{"month": 1, "dti": 0.3}]))

    def test_negative_consecutive_missed_raises(self):
        with self.assertRaises(ValueError):
            build_case(_base_case([{"month": 1, "income": 10000, "dti": 0.3, "consecutive_missed": -1}]))

    def test_defaults_applied(self):
        case = build_case(_base_case([{"month": 1, "income": 10000, "dti": 0.3}]))
        self.assertEqual(case.tenure_months, 24)
        self.assertAlmostEqual(case.typical_monthly_income, 10000)

    def test_valid_case_with_arrears_and_original_emi(self):
        case = build_case({
            "borrower_id": "b1", "borrower_type": "trader", "original_emi": 2000,
            "months": [{"month": 1, "income": 10000, "dti": 0.3, "arrears": 500}],
        })
        self.assertEqual(case.original_emi, 2000)


class TestObservationConversion(unittest.TestCase):
    def test_dti_is_passed_through_directly(self):
        case = build_case(_base_case([{"month": 1, "income": 10000, "dti": 0.42}]))
        _, obs = next(iter_observations(case))
        dti_index = FEATURE_NAMES.index("dti")
        self.assertAlmostEqual(float(obs[dti_index]), 0.42, places=4)

    def test_income_history_accumulates_across_months(self):
        case = build_case(_base_case([
            {"month": 1, "income": 15000, "dti": 0.3},
            {"month": 2, "income": 13000, "dti": 0.35},
            {"month": 3, "income": 9000, "dti": 0.5},
        ]))
        results = list(iter_observations(case))
        income_cv_idx = FEATURE_NAMES.index("income_cv")
        # Volatility should be nonzero once incomes actually vary.
        self.assertEqual(float(results[0][1][income_cv_idx]), 0.0)
        self.assertGreater(float(results[2][1][income_cv_idx]), 0.0)

    def test_momentum_zero_before_three_months(self):
        case = build_case(_base_case([
            {"month": 1, "income": 15000, "dti": 0.3},
            {"month": 2, "income": 13000, "dti": 0.35},
        ]))
        results = list(iter_observations(case))
        momentum_idx = FEATURE_NAMES.index("momentum")
        self.assertEqual(float(results[0][1][momentum_idx]), 0.0)
        self.assertEqual(float(results[1][1][momentum_idx]), 0.0)

    def test_momentum_negative_for_declining_income(self):
        case = build_case(_base_case([
            {"month": 1, "income": 15000, "dti": 0.3},
            {"month": 2, "income": 12000, "dti": 0.4},
            {"month": 3, "income": 9000, "dti": 0.5},
        ]))
        results = list(iter_observations(case))
        momentum_idx = FEATURE_NAMES.index("momentum")
        self.assertLess(float(results[2][1][momentum_idx]), 0.0)

    def test_deterministic_across_runs(self):
        case = build_case(_base_case([
            {"month": 1, "income": 15000, "dti": 0.3},
            {"month": 2, "income": 8000, "dti": 0.6, "savings_buffer": 500, "consecutive_missed": 1},
        ]))
        run1 = [obs.tolist() for _, obs in iter_observations(case)]
        run2 = [obs.tolist() for _, obs in iter_observations(case)]
        self.assertEqual(run1, run2)

    def test_arrears_ratio_uses_original_emi(self):
        case = build_case({
            "borrower_id": "b1", "borrower_type": "trader", "original_emi": 2000,
            "months": [{"month": 1, "income": 10000, "dti": 0.5, "arrears": 1000}],
        })
        _, obs = next(iter_observations(case))
        arrears_idx = FEATURE_NAMES.index("arrears_ratio")
        self.assertAlmostEqual(float(obs[arrears_idx]), 0.5, places=4)

    def test_consecutive_missed_scaled(self):
        case = build_case(_base_case([{"month": 1, "income": 10000, "dti": 0.3, "consecutive_missed": 2}]))
        _, obs = next(iter_observations(case))
        idx = FEATURE_NAMES.index("consecutive_missed")
        self.assertAlmostEqual(float(obs[idx]), 2 / 5.0, places=4)

    def test_months_remaining_frac_defaults_from_tenure(self):
        case = build_case({
            "borrower_id": "b1", "borrower_type": "trader", "tenure_months": 12,
            "months": [
                {"month": 1, "income": 10000, "dti": 0.3},
                {"month": 2, "income": 10000, "dti": 0.3},
            ],
        })
        results = list(iter_observations(case))
        idx = FEATURE_NAMES.index("months_remaining_frac")
        self.assertAlmostEqual(float(results[0][1][idx]), 12 / 12.0, places=4)
        self.assertAlmostEqual(float(results[1][1][idx]), 11 / 12.0, places=4)

    def test_explicit_months_remaining_overrides_default(self):
        case = build_case({
            "borrower_id": "b1", "borrower_type": "trader", "tenure_months": 12,
            "months": [{"month": 1, "income": 10000, "dti": 0.3, "months_remaining": 1}],
        })
        _, obs = next(iter_observations(case))
        idx = FEATURE_NAMES.index("months_remaining_frac")
        self.assertAlmostEqual(float(obs[idx]), 1 / 12.0, places=4)


if __name__ == "__main__":
    unittest.main()
