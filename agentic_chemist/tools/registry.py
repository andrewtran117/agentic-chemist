"""Tool registry: maps tool names to implementations and generates Anthropic API tool schemas."""

import inspect
import typing
from dataclasses import fields

from agentic_chemist.schemas import MoleculeInfo, Prediction
from agentic_chemist.tools.parse import parse_molecule
from agentic_chemist.tools.similarity import compute_similarity
from agentic_chemist.tools.admet import predict_admet, ENDPOINTS

# Python type -> JSON schema type
_TYPE_MAP = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}


def _python_type_to_json(annotation) -> dict:
    """Convert a Python type annotation to a JSON schema fragment."""
    if annotation in _TYPE_MAP:
        return {"type": _TYPE_MAP[annotation]}
    # Handle Optional / Union with None
    origin = typing.get_origin(annotation)
    if origin is typing.Union:
        args = [a for a in typing.get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            return _python_type_to_json(args[0])
    return {"type": "string"}


def _build_schema(func, description: str, param_overrides: dict | None = None) -> dict:
    """Build an Anthropic tool schema from a function's signature."""
    sig = inspect.signature(func)
    hints = typing.get_type_hints(func)

    properties = {}
    required = []

    for name, param in sig.parameters.items():
        if name in ("self", "cls"):
            continue
        prop = _python_type_to_json(hints.get(name, str))
        if param_overrides and name in param_overrides:
            prop.update(param_overrides[name])
        properties[name] = prop
        if param.default is inspect.Parameter.empty:
            required.append(name)

    return {
        "name": func.__name__,
        "description": description,
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }


# --- Registry ---

TOOL_REGISTRY = {
    "parse_molecule": {
        "fn": parse_molecule,
        "schema": _build_schema(
            parse_molecule,
            "Parse a SMILES string and return molecular descriptors: molecular weight, "
            "logP, hydrogen bond donors/acceptors, TPSA, rotatable bonds, QED, "
            "Lipinski rule-of-five pass/fail, molecular formula, and ring count.",
        ),
    },
    "compute_similarity": {
        "fn": compute_similarity,
        "schema": _build_schema(
            compute_similarity,
            "Compute Tanimoto similarity between two molecules using Morgan fingerprints. "
            "Returns a float in [0, 1]. Use this to compare structural similarity or cluster molecules.",
        ),
    },
    "predict_admet": {
        "fn": predict_admet,
        "schema": _build_schema(
            predict_admet,
            "Predict an ADMET property for a molecule. Returns the predicted value, "
            "units, model ID, and an applicability domain flag indicating whether the "
            "molecule is within the training distribution. Available endpoints: "
            + ", ".join(ENDPOINTS.keys()) + ".",
            param_overrides={
                "endpoint": {"enum": list(ENDPOINTS.keys())},
            },
        ),
    },
}


def get_tool_schemas() -> list[dict]:
    """Return the list of tool schemas for the Anthropic API."""
    return [entry["schema"] for entry in TOOL_REGISTRY.values()]


def call_tool(name: str, kwargs: dict):
    """Call a registered tool by name. Returns the result or raises."""
    if name not in TOOL_REGISTRY:
        raise ValueError(f"Unknown tool: {name}")
    return TOOL_REGISTRY[name]["fn"](**kwargs)
