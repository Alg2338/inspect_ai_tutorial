"""Offline behavior checks for Tutorial 1's notebook-only results route."""

import ast
import contextlib
import io
import json
import os
import sys
import tempfile
import types
import unittest
import zipfile
from pathlib import Path
from unittest.mock import Mock, patch

from inspect_ai.log import (
    EvalConfig,
    EvalDataset,
    EvalLog,
    EvalMetric,
    EvalResults,
    EvalSample,
    EvalScore,
    EvalSpec,
)
from inspect_ai.model import ModelOutput
from inspect_ai.scorer import Score

from check_notebooks import prepare

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = json.loads(
    (ROOT / "inspect_ai_tutorial_1_basics.ipynb").read_text(encoding="utf-8")
)
CELLS = {cell["id"]: "".join(cell["source"]) for cell in NOTEBOOK["cells"]}


def sample_log() -> EvalLog:
    return EvalLog(
        status="success",
        eval=EvalSpec(
            created="2026-10-28T00:00:00Z",
            task="offline_results",
            model="mockllm/model",
            dataset=EvalDataset(samples=1),
            config=EvalConfig(),
        ),
        results=EvalResults(
            scores=[
                EvalScore(
                    name="match",
                    scorer="match",
                    metrics={"accuracy": EvalMetric(name="accuracy", value=1.0)},
                )
            ]
        ),
        samples=[
            EvalSample(
                id=1,
                epoch=1,
                input="2+2=",
                target="4",
                output=ModelOutput(completion="4"),
                scores={"match": Score(value="C")},
                metadata={"difficulty": "easy"},
            )
        ],
    )


class InlineResultsTests(unittest.TestCase):
    def display(self, log: EvalLog) -> str:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exec(
                compile(CELLS["inline-results"], "<inline-results>", "exec"),
                {"results": [log]},
            )
        return output.getvalue()

    def test_displays_metrics_and_sample_without_a_viewer(self) -> None:
        output = self.display(sample_log())
        for expected in (
            "Status: success",
            "accuracy",
            "1.0",
            "Input: 2+2=",
            "Target: 4",
            "Response: 4",
            "'match': 'C'",
            "'difficulty': 'easy'",
        ):
            with self.subTest(expected=expected):
                self.assertIn(expected, output)

    def test_display_limits_sample_preview_to_three(self) -> None:
        log = sample_log()
        log.samples = [
            log.samples[0].model_copy(update={"input": f"Question {index}"})
            for index in range(1, 5)
        ]
        output = self.display(log)
        self.assertIn("Question 3", output)
        self.assertNotIn("Question 4", output)

    def test_missing_results_samples_and_scores_do_not_crash(self) -> None:
        log = sample_log()
        log.results = None
        log.samples[0].scores = None
        self.assertIn("Scores: {}", self.display(log))
        log.samples = None
        output = self.display(log)
        self.assertIn("No aggregate scores available.", output)
        self.assertIn("No samples available.", output)

    def test_existing_evaluations_are_captured_not_repeated(self) -> None:
        marker = [sample_log()]
        initial_eval = Mock(return_value=marker)
        namespace = {
            "eval": initial_eval,
            "hello_model": object(),
            "MODEL": "mockllm/model",
        }
        with contextlib.redirect_stdout(io.StringIO()):
            exec(CELLS["d1087d87"], namespace)
        initial_eval.assert_called_once()
        self.assertIs(namespace["results"], marker)

        # Only run model-result capture/display statements, not student exercises.
        source = CELLS["a707981e-2718-456b-8d47-04a321e87545"]
        tree = ast.parse(prepare(source))
        statements = []
        for node in tree.body:
            value = getattr(node, "value", None)
            if (
                isinstance(node, (ast.Assign, ast.Expr))
                and isinstance(value, ast.Call)
                and isinstance(value.func, ast.Name)
                and value.func.id in {"eval", "show_results", "print"}
            ):
                statements.append(node)
        biased, unbiased = [sample_log()], [sample_log()]
        run_eval = Mock(side_effect=[biased, unbiased])
        display_namespace = {"results": marker}
        with contextlib.redirect_stdout(io.StringIO()):
            exec(CELLS["inline-results"], display_namespace)
        namespace = {
            "eval": run_eval,
            "show_results": display_namespace["show_results"],
            "position_bias_task": lambda samples: samples,
            "samples_biased": object(),
            "samples_unbiased": object(),
            "MODEL": "mockllm/model",
        }
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exec(
                compile(
                    ast.Module(body=statements, type_ignores=[]),
                    "<capstone-results>",
                    "exec",
                ),
                namespace,
            )
        self.assertEqual(run_eval.call_count, 2)
        self.assertIs(namespace["biased_results"], biased)
        self.assertIs(namespace["unbiased_results"], unbiased)
        self.assertIn("Biased run:", output.getvalue())
        self.assertIn("Unbiased run:", output.getvalue())
        self.assertEqual(output.getvalue().count("'accuracy': 1.0"), 2)


@unittest.skipUnless("30e1c4ba" in CELLS, "Colab-only download cell")
class ColabDownloadTests(unittest.TestCase):
    def enabled_code(self) -> types.CodeType:
        tree = ast.parse(CELLS["30e1c4ba"])
        toggle = next(
            node
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "DOWNLOAD_LOGS"
                for target in node.targets
            )
        )
        self.assertIs(toggle.value.value, False)
        toggle.value = ast.Constant(value=True)
        return compile(ast.fix_missing_locations(tree), "<colab-download>", "exec")

    def test_default_does_not_import_colab_or_download(self) -> None:
        with patch.dict(sys.modules, {"google.colab": None}):
            exec(CELLS["30e1c4ba"], {})

    def run_enabled(self, with_log: bool) -> str:
        google = types.ModuleType("google")
        colab = types.ModuleType("google.colab")
        colab.files = Mock()
        google.colab = colab
        previous = Path.cwd()
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            os.chdir(directory)
            try:
                if with_log:
                    Path("logs").mkdir()
                    Path("logs/example.eval").write_bytes(b"offline fixture")
                Path(".env").write_text("EXCLUDED=true", encoding="utf-8")
                with patch.dict(
                    sys.modules, {"google": google, "google.colab": colab}
                ):
                    with contextlib.redirect_stdout(output):
                        exec(self.enabled_code(), {})
                if with_log:
                    colab.files.download.assert_called_once()
                    with zipfile.ZipFile("inspect_logs.zip") as archive:
                        self.assertEqual(
                            archive.read("logs/example.eval"), b"offline fixture"
                        )
                        self.assertNotIn(".env", archive.namelist())
                else:
                    colab.files.download.assert_not_called()
                    self.assertFalse(Path("inspect_logs.zip").exists())
            finally:
                os.chdir(previous)
        return output.getvalue()

    def test_opt_in_zip_matches_local_viewer_path(self) -> None:
        self.run_enabled(with_log=True)

    def test_missing_logs_prints_guidance_without_exiting(self) -> None:
        self.assertIn("No .eval logs found", self.run_enabled(with_log=False))


if __name__ == "__main__":
    unittest.main()
