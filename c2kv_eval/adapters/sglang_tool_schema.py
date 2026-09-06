from __future__ import annotations

from typing import Any, Mapping, Sequence


def normalize_tools_for_sglang(
    tools: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Match SGLang's Pydantic dump shape for valid OpenAI tools."""
    normalized: list[dict[str, Any]] = []
    for tool in tools:
        function = tool["function"]
        normalized.append(
            {
                "type": tool.get("type", "function"),
                "function": {
                    "description": function.get("description"),
                    "name": function["name"],
                    "parameters": function.get("parameters"),
                    "strict": function.get("strict", False),
                },
            }
        )
    return normalized
