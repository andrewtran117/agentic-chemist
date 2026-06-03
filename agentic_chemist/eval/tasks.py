"""Eval task definitions: curated tasks with expected tool calls and ground truth."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvalTask:
    id: str
    prompt: str
    expected_tool_calls: list[dict]  # [{"tool": name, "args": {key: val}}]
    ground_truth: dict  # expected values in the final answer
    category: str  # "parse", "similarity", "admet", "multi_tool", "abstention"
    description: str = ""


EVAL_TASKS = [
    # --- parse_molecule tasks ---
    EvalTask(
        id="parse_lipinski_pass",
        prompt="Does aspirin (CC(=O)Oc1ccccc1C(=O)O) satisfy Lipinski's Rule of Five?",
        expected_tool_calls=[
            {"tool": "parse_molecule", "args": {"smiles": "CC(=O)Oc1ccccc1C(=O)O"}},
        ],
        ground_truth={"lipinski_pass": True, "lipinski_violations": 0},
        category="parse",
        description="Single molecule Lipinski check — should pass.",
    ),
    EvalTask(
        id="parse_lipinski_fail",
        prompt=(
            "Does atorvastatin "
            "(CC(C)c1n(CC[C@@H](O)C[C@@H](O)CC(=O)[O-])c(-c2ccccc2)c(-c2ccc(F)cc2)c1C(=O)Nc1ccccc1) "
            "satisfy Lipinski's Rule of Five?"
        ),
        expected_tool_calls=[
            {"tool": "parse_molecule", "args": {"smiles": "CC(C)c1n(CC[C@@H](O)C[C@@H](O)CC(=O)[O-])c(-c2ccccc2)c(-c2ccc(F)cc2)c1C(=O)Nc1ccccc1"}},
        ],
        ground_truth={"lipinski_pass": False, "lipinski_violations": 1},
        category="parse",
        description="Single molecule Lipinski check — should fail (MW > 500).",
    ),
    EvalTask(
        id="parse_compare_qed",
        prompt=(
            "Compare the drug-likeness (QED) of aspirin (CC(=O)Oc1ccccc1C(=O)O) "
            "and ibuprofen (CC(C)Cc1ccc(C(C)C(=O)O)cc1). Which has higher QED?"
        ),
        expected_tool_calls=[
            {"tool": "parse_molecule", "args": {"smiles": "CC(=O)Oc1ccccc1C(=O)O"}},
            {"tool": "parse_molecule", "args": {"smiles": "CC(C)Cc1ccc(C(C)C(=O)O)cc1"}},
        ],
        ground_truth={"higher_qed": "ibuprofen", "aspirin_qed": 0.55, "ibuprofen_qed": 0.822},
        category="parse",
        description="Compare QED of two molecules — agent must parse both.",
    ),

    # --- compute_similarity tasks ---
    EvalTask(
        id="sim_identical",
        prompt="How similar are these two SMILES? c1ccccc1 and C1=CC=CC=C1",
        expected_tool_calls=[
            {"tool": "compute_similarity", "args": {"smiles_a": "c1ccccc1", "smiles_b": "C1=CC=CC=C1"}},
        ],
        ground_truth={"similarity": 1.0, "are_identical": True},
        category="similarity",
        description="Same molecule in different notation — should be 1.0.",
    ),
    EvalTask(
        id="sim_dissimilar",
        prompt=(
            "How structurally similar are ethanol (CCO) and "
            "ibuprofen (CC(C)Cc1ccc(C(C)C(=O)O)cc1)?"
        ),
        expected_tool_calls=[
            {"tool": "compute_similarity", "args": {"smiles_a": "CCO", "smiles_b": "CC(C)Cc1ccc(C(C)C(=O)O)cc1"}},
        ],
        ground_truth={"similarity": 0.107, "are_similar": False},
        category="similarity",
        description="Very different molecules — low Tanimoto expected.",
    ),

    # --- predict_admet tasks ---
    EvalTask(
        id="admet_solubility",
        prompt="Predict the aqueous solubility of aspirin (CC(=O)Oc1ccccc1C(=O)O).",
        expected_tool_calls=[
            {"tool": "predict_admet", "args": {"smiles": "CC(=O)Oc1ccccc1C(=O)O", "endpoint": "solubility"}},
        ],
        ground_truth={"endpoint": "solubility", "value_range": [-3.0, -0.5], "applicability": "in_domain"},
        category="admet",
        description="Solubility prediction for a common drug — in domain.",
    ),
    EvalTask(
        id="admet_herg",
        prompt="Assess the hERG cardiac risk for aspirin (CC(=O)Oc1ccccc1C(=O)O).",
        expected_tool_calls=[
            {"tool": "predict_admet", "args": {"smiles": "CC(=O)Oc1ccccc1C(=O)O", "endpoint": "herg"}},
        ],
        ground_truth={"endpoint": "herg", "value_range": [0.0, 1.0], "applicability": "in_domain"},
        category="admet",
        description="hERG prediction — agent must pick the right endpoint.",
    ),

    # --- abstention task ---
    EvalTask(
        id="admet_ood_abstention",
        prompt=(
            "Predict the solubility of ferrocene ([Fe+2].[cH-]1cccc1.[cH-]1cccc1). "
            "How confident should we be in this prediction?"
        ),
        expected_tool_calls=[
            {"tool": "predict_admet", "args": {"smiles": "[Fe+2].[cH-]1cccc1.[cH-]1cccc1", "endpoint": "solubility"}},
        ],
        ground_truth={"applicability": "borderline", "should_warn": True},
        category="abstention",
        description="OOD organometallic — agent should flag low confidence.",
    ),

    # --- multi-tool tasks ---
    EvalTask(
        id="multi_screen_lipinski_solubility",
        prompt=(
            "Screen aspirin (CC(=O)Oc1ccccc1C(=O)O) and metformin (CN(C)C(=N)NC(=N)N). "
            "For each: check Lipinski compliance and predict solubility. "
            "Which is more soluble?"
        ),
        expected_tool_calls=[
            {"tool": "parse_molecule", "args": {"smiles": "CC(=O)Oc1ccccc1C(=O)O"}},
            {"tool": "parse_molecule", "args": {"smiles": "CN(C)C(=N)NC(=N)N"}},
            {"tool": "predict_admet", "args": {"smiles": "CC(=O)Oc1ccccc1C(=O)O", "endpoint": "solubility"}},
            {"tool": "predict_admet", "args": {"smiles": "CN(C)C(=N)NC(=N)N", "endpoint": "solubility"}},
        ],
        ground_truth={
            "both_lipinski_pass": True,
            "more_soluble": "metformin",
            "aspirin_solubility": -1.86,
            "metformin_solubility": -0.50,
        },
        category="multi_tool",
        description="Multi-molecule screen requiring parse + ADMET for each.",
    ),
    EvalTask(
        id="multi_full_profile",
        prompt=(
            "Give me a full safety profile of ibuprofen (CC(C)Cc1ccc(C(C)C(=O)O)cc1): "
            "molecular properties, predicted solubility, and hERG risk."
        ),
        expected_tool_calls=[
            {"tool": "parse_molecule", "args": {"smiles": "CC(C)Cc1ccc(C(C)C(=O)O)cc1"}},
            {"tool": "predict_admet", "args": {"smiles": "CC(C)Cc1ccc(C(C)C(=O)O)cc1", "endpoint": "solubility"}},
            {"tool": "predict_admet", "args": {"smiles": "CC(C)Cc1ccc(C(C)C(=O)O)cc1", "endpoint": "herg"}},
        ],
        ground_truth={
            "lipinski_pass": True,
            "endpoints_covered": ["solubility", "herg"],
        },
        category="multi_tool",
        description="Full profile requiring all three tool types.",
    ),
]


def get_tasks_by_category(category: str) -> list[EvalTask]:
    return [t for t in EVAL_TASKS if t.category == category]


def get_all_tasks() -> list[EvalTask]:
    return EVAL_TASKS
