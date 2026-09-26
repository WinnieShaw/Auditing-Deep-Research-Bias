import unittest

from cess_audit import estimate_episode, evaluate


def episode():
    return {
        "task_id": "t1",
        "ranking": "supporting_first",
        "target": 0.25,
        "outcome_regression": 0.2,
        "searches": [
            {
                "candidate_count": 4,
                "selected_probability": 0.5,
                "continuation_probability": 0.5,
                "selected_outcome": 0.4,
                "selected_outcome_prediction": 0.3,
            },
        ],
    }


class AuditTests(unittest.TestCase):
    def test_logged_support_is_used_without_epsilon(self):
        estimate = estimate_episode(episode())
        # 1/(4*.5*.5) = 1; DR = .2 + (.4-.3) = .3.
        self.assertAlmostEqual(estimate["dr_raw"], 0.3)
        self.assertAlmostEqual(estimate["cess"], 0.25)

    def test_zero_probability_is_rejected(self):
        bad = episode()
        bad["searches"][0]["selected_probability"] = 0
        with self.assertRaises(ValueError):
            estimate_episode(bad)

    def test_horizon_and_global_anchor_are_explicit(self):
        row = estimate_episode(episode(), horizon=1, global_mean=0.8)
        self.assertAlmostEqual(row["opened_to_global"], 0.6)

    def test_task_macro_ranking_sensitivity(self):
        other = episode()
        other["ranking"] = "opposing_first"
        rows, summary = evaluate([episode(), other])
        self.assertEqual(len(rows), 2)
        self.assertIn("cess", summary["metrics"])
        self.assertGreaterEqual(summary["metrics"]["cess"]["ranking_sensitivity"], 0)


if __name__ == "__main__":
    unittest.main()

