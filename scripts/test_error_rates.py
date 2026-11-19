"""Offline checks for Assignment 2's rate formulas, without solving its TODO.

Run: python scripts/test_error_rates.py
"""
import contextlib
import io
import json
from math import isnan
from pathlib import Path
from types import SimpleNamespace
import unittest


NOTEBOOK = Path(__file__).resolve().parents[1] / "inspect_ai_tutorial_3_llm_judge.ipynb"


def run_cell(total, **counts):
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
        and "def compute_error_rates(" in "".join(cell["source"])
    )
    # Supply the counters learners will implement; exercise the real return and checks.
    namespace = {
        "EvalLog": object,
        "results": [SimpleNamespace(samples=[None] * total)],
        **counts,
    }
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(source, str(NOTEBOOK), "exec"), namespace)
    return namespace["rates"]


class ErrorRatesTests(unittest.TestCase):
    def test_conditional_denominators(self):
        rates = run_cell(
            12, clf_fp=1, clf_fn=2, clf_fail=3, clf_neg=3, clf_pos=9,
            judge_fp=2, judge_fn=1, judge_fail=1, judge_other_rate=2,
            judge_neg=6, judge_pos=3,
        )
        self.assertEqual(rates, {
            "clf_fp_rate": 1 / 3, "clf_fn_rate": 2 / 9,
            "clf_failure_rate": 3 / 12,
            "judge_fp_rate": 2 / 6, "judge_fn_rate": 1 / 3,
            "judge_failure_rate": 1 / 12, "judge_other_rate": 2 / 12,
        })

    def test_classifier_conditional_rates_need_not_sum_to_one(self):
        rates = run_cell(
            12, clf_fp=3, clf_fn=9, clf_fail=0, clf_neg=3, clf_pos=9,
            judge_fp=0, judge_fn=12, judge_fail=0, judge_other_rate=0,
            judge_neg=0, judge_pos=12,
        )
        self.assertEqual(rates["clf_fp_rate"], 1.0)
        self.assertEqual(rates["clf_fn_rate"], 1.0)

    def test_judge_conditional_rates_need_not_sum_to_one(self):
        rates = run_cell(
            12, clf_fp=3, clf_fn=3, clf_fail=0, clf_neg=6, clf_pos=6,
            judge_fp=6, judge_fn=6, judge_fail=0, judge_other_rate=0,
            judge_neg=6, judge_pos=6,
        )
        self.assertEqual(rates["judge_fp_rate"], 1.0)
        self.assertEqual(rates["judge_fn_rate"], 1.0)

    def test_absent_classes_have_undefined_conditional_rates(self):
        rates = run_cell(
            3, clf_fp=0, clf_fn=0, clf_fail=3, clf_neg=3, clf_pos=0,
            judge_fp=0, judge_fn=0, judge_fail=0, judge_other_rate=3,
            judge_neg=0, judge_pos=0,
        )
        self.assertEqual(rates["clf_fp_rate"], 0.0)
        self.assertEqual(rates["clf_failure_rate"], 1.0)
        for name in ("clf_fn_rate", "judge_fp_rate", "judge_fn_rate"):
            self.assertTrue(isnan(rates[name]), name)

    def test_no_negative_classifier_class(self):
        rates = run_cell(
            3, clf_fp=0, clf_fn=1, clf_fail=1, clf_neg=0, clf_pos=3,
            judge_fp=0, judge_fn=1, judge_fail=1, judge_other_rate=0,
            judge_neg=1, judge_pos=1,
        )
        self.assertTrue(isnan(rates["clf_fp_rate"]))
        self.assertEqual(rates["clf_fn_rate"], 1 / 3)

    def test_empty_log_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least one sample"):
            run_cell(0)


if __name__ == "__main__":
    unittest.main()
