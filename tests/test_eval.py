"""Tests for eval scorers."""

import pytest
from agentic_chemist.agent.trace import Trace
from agentic_chemist.eval.tasks import EvalTask, get_all_tasks, get_tasks_by_category
from agentic_chemist.eval.scorers import (
    score_trajectory,
    score_grounding,
    score_abstention,
    evaluate,
)


def _make_task():
    """Helper: a simple parse task."""
    return EvalTask(
        id="test",
        prompt="Check Lipinski for aspirin",
        expected_tool_calls=[
            {"tool": "parse_molecule", "args": {"smiles": "CC(=O)Oc1ccccc1C(=O)O"}},
        ],
        ground_truth={"lipinski_pass": True},
        category="parse",
    )


class TestTrajectoryScorer:

    def test_perfect_trajectory(self):
        task = _make_task()
        trace = Trace(task=task.prompt)
        trace.add_tool_call(
            "parse_molecule",
            {"smiles": "CC(=O)Oc1ccccc1C(=O)O"},
            {"lipinski_pass": True},
            5.0,
        )
        score = score_trajectory(trace, task)
        assert score.tool_recall == 1.0
        assert score.tool_precision == 1.0
        assert score.redundant_calls == 0
        assert score.missing_tools == []

    def test_missing_tool_call(self):
        task = _make_task()
        trace = Trace(task=task.prompt)
        # Agent didn't call anything
        score = score_trajectory(trace, task)
        assert score.tool_recall == 0.0
        assert score.missing_tools == ["parse_molecule"]

    def test_extra_tool_call(self):
        task = _make_task()
        trace = Trace(task=task.prompt)
        trace.add_tool_call(
            "parse_molecule",
            {"smiles": "CC(=O)Oc1ccccc1C(=O)O"},
            {"lipinski_pass": True},
            5.0,
        )
        trace.add_tool_call(
            "compute_similarity",
            {"smiles_a": "CCO", "smiles_b": "CCCO"},
            0.5,
            3.0,
        )
        score = score_trajectory(trace, task)
        assert score.tool_recall == 1.0
        assert score.redundant_calls == 1
        assert score.extra_tools == ["compute_similarity"]

    def test_wrong_tool(self):
        task = _make_task()
        trace = Trace(task=task.prompt)
        trace.add_tool_call(
            "predict_admet",
            {"smiles": "CC(=O)Oc1ccccc1C(=O)O", "endpoint": "solubility"},
            {"value": -1.5},
            5.0,
        )
        score = score_trajectory(trace, task)
        assert score.tool_recall == 0.0
        assert "parse_molecule" in score.missing_tools

    def test_multi_tool_task(self):
        task = EvalTask(
            id="multi",
            prompt="Screen two molecules",
            expected_tool_calls=[
                {"tool": "parse_molecule", "args": {"smiles": "CCO"}},
                {"tool": "predict_admet", "args": {"smiles": "CCO", "endpoint": "solubility"}},
            ],
            ground_truth={},
            category="multi_tool",
        )
        trace = Trace(task=task.prompt)
        trace.add_tool_call("parse_molecule", {"smiles": "CCO"}, {}, 1.0)
        trace.add_tool_call("predict_admet", {"smiles": "CCO", "endpoint": "solubility"}, {}, 1.0)
        score = score_trajectory(trace, task)
        assert score.tool_recall == 1.0
        assert score.tool_precision == 1.0


class TestGroundingScorer:

    def test_grounded_response(self):
        trace = Trace(task="test")
        trace.add_tool_call(
            "parse_molecule",
            {"smiles": "CCO"},
            {"molecular_weight": 46.04, "lipinski_pass": True},
            1.0,
        )
        trace.finalize("The molecular weight is 46.04 and it passes Lipinski.")
        score = score_grounding(trace)
        assert score.tools_referenced_in_response >= 1
        assert score.has_ungrounded_claims is False

    def test_ungrounded_response(self):
        trace = Trace(task="test")
        trace.add_tool_call(
            "parse_molecule",
            {"smiles": "CCO"},
            {"molecular_weight": 46.04},
            1.0,
        )
        trace.finalize("This molecule is very toxic and should not be used.")
        score = score_grounding(trace)
        # Response doesn't reference the actual tool output
        assert score.has_ungrounded_claims is True

    def test_no_response(self):
        trace = Trace(task="test")
        trace.add_tool_call("parse_molecule", {"smiles": "CCO"}, {}, 1.0)
        score = score_grounding(trace)
        assert score.has_ungrounded_claims is True


class TestAbstentionScorer:

    def test_correct_warning(self):
        task = EvalTask(
            id="ood",
            prompt="Predict solubility of ferrocene",
            expected_tool_calls=[],
            ground_truth={"should_warn": True},
            category="abstention",
        )
        trace = Trace(task=task.prompt)
        trace.finalize("The prediction is borderline — this molecule is outside the training domain.")
        score = score_abstention(trace, task)
        assert score is not None
        assert score.did_warn is True
        assert score.correct is True

    def test_missing_warning(self):
        task = EvalTask(
            id="ood",
            prompt="Predict solubility of ferrocene",
            expected_tool_calls=[],
            ground_truth={"should_warn": True},
            category="abstention",
        )
        trace = Trace(task=task.prompt)
        trace.finalize("The predicted solubility is -2.86 logS.")
        score = score_abstention(trace, task)
        assert score.did_warn is False
        assert score.correct is False

    def test_no_abstention_needed(self):
        task = _make_task()
        trace = Trace(task=task.prompt)
        trace.finalize("Lipinski pass.")
        score = score_abstention(trace, task)
        assert score is None  # not an abstention task


class TestFullEvaluate:

    def test_evaluate_returns_all_scores(self):
        task = _make_task()
        trace = Trace(task=task.prompt)
        trace.add_tool_call(
            "parse_molecule",
            {"smiles": "CC(=O)Oc1ccccc1C(=O)O"},
            {"lipinski_pass": True, "molecular_weight": 180.04},
            5.0,
        )
        trace.finalize("Aspirin has MW 180.04 and passes Lipinski.")
        result = evaluate(trace, task)
        assert result.task_id == "test"
        assert result.trajectory.tool_recall == 1.0
        assert result.grounding.has_ungrounded_claims is False
        assert result.abstention is None

    def test_evaluate_to_dict(self):
        task = _make_task()
        trace = Trace(task=task.prompt)
        trace.finalize("done")
        result = evaluate(trace, task)
        d = result.to_dict()
        assert "trajectory" in d
        assert "grounding" in d


class TestTaskDefinitions:

    def test_all_tasks_have_required_fields(self):
        for task in get_all_tasks():
            assert task.id
            assert task.prompt
            assert task.category
            assert isinstance(task.expected_tool_calls, list)
            assert isinstance(task.ground_truth, dict)

    def test_categories_covered(self):
        categories = {t.category for t in get_all_tasks()}
        assert "parse" in categories
        assert "similarity" in categories
        assert "admet" in categories
        assert "abstention" in categories
        assert "multi_tool" in categories

    def test_get_by_category(self):
        parse_tasks = get_tasks_by_category("parse")
        assert all(t.category == "parse" for t in parse_tasks)
        assert len(parse_tasks) >= 2
