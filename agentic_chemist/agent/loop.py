"""Agent loop: plan → call tool → observe → reflect → synthesize."""

import json
import time

import anthropic

from agentic_chemist.agent.trace import Trace
from agentic_chemist.tools.registry import get_tool_schemas, call_tool

DEFAULT_MODEL = "claude-sonnet-4-20250514"
MAX_STEPS = 15
SYSTEM_PROMPT = """You are a cheminformatics assistant. You help scientists analyze molecules by calling tools to compute molecular properties, predict ADMET endpoints, and compare molecular structures.

Rules:
1. ALWAYS use tools to get data. Never guess or hallucinate molecular properties.
2. Every claim in your final report MUST be backed by a tool call result from this conversation.
3. If a prediction has an applicability flag of "borderline" or "out_of_domain", explicitly warn the user that the prediction may be unreliable.
4. Be concise and structured in your final report.
5. When comparing molecules, use compute_similarity to quantify structural relationships.
6. For ADMET predictions, always note the model_id and applicability_flag.

When you have gathered enough information, synthesize a final report in markdown format with:
- A summary of findings
- A table of results where appropriate
- Any warnings about out-of-domain predictions
- Clear links between claims and the tool calls that produced them"""


def run_agent(
    task: str,
    model: str = DEFAULT_MODEL,
    max_steps: int = MAX_STEPS,
    verbose: bool = False,
) -> Trace:
    """Run the agent loop on a natural-language task.

    Returns a Trace with every tool call, token usage, and the final response.
    """
    client = anthropic.Anthropic()
    tools = get_tool_schemas()
    trace = Trace(task=task)

    messages = [{"role": "user", "content": task}]

    for step in range(max_steps):
        if verbose:
            print(f"\n--- Step {step + 1} ---")

        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=tools,
            messages=messages,
        )

        # Track token usage
        trace.add_usage(response.usage.input_tokens, response.usage.output_tokens)

        # Check if the model wants to use tools
        if response.stop_reason == "end_turn":
            # Model is done — extract final text
            final_text = ""
            for block in response.content:
                if block.type == "text":
                    final_text += block.text
            trace.finalize(final_text)
            if verbose:
                print(f"\n--- Agent done in {trace.step_count} tool calls ---")
            return trace

        # Process tool uses
        assistant_content = response.content
        tool_results = []

        for block in assistant_content:
            if block.type == "text" and verbose:
                print(f"Agent: {block.text}")
            elif block.type == "tool_use":
                tool_name = block.name
                tool_input = block.input

                if verbose:
                    print(f"Tool call: {tool_name}({json.dumps(tool_input)})")

                # Execute the tool
                t0 = time.time()
                try:
                    result = call_tool(tool_name, tool_input)
                    latency = (time.time() - t0) * 1000
                    # Serialize result
                    if hasattr(result, "to_dict"):
                        result_data = result.to_dict()
                    elif result is None:
                        result_data = {"error": "Invalid input — tool returned None"}
                    else:
                        result_data = result

                    trace.add_tool_call(tool_name, tool_input, result_data, latency)

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result_data, default=str),
                    })
                except Exception as e:
                    latency = (time.time() - t0) * 1000
                    trace.add_tool_call(tool_name, tool_input, None, latency, error=str(e))
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps({"error": str(e)}),
                        "is_error": True,
                    })

                if verbose:
                    print(f"  -> {json.dumps(result_data if 'result_data' in dir() else str(e), default=str)[:200]}")

        # Append assistant message + tool results
        messages.append({"role": "assistant", "content": assistant_content})
        messages.append({"role": "user", "content": tool_results})

    # Hit step limit
    trace.finalize("Error: agent exceeded maximum step budget.")
    return trace
