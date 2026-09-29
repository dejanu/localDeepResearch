"""Clarifier: decides whether the research question is underspecified."""
import json
import anthropic

from app.config import config

client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

CLARIFIER_TOOL = {
    "name": "emit_clarification_check",
    "description": "Report whether the question needs clarification before research can begin.",
    "input_schema": {
        "type": "object",
        "properties": {
            "needs_clarification": {"type": "boolean"},
            "questions": {
                "type": "array",
                "items": {"type": "string"},
                "description": "1-3 clarifying questions, only if needs_clarification is true",
            },
        },
        "required": ["needs_clarification"],
    },
}

SYSTEM_PROMPT = """You check whether a research question is well-specified \
enough to research productively. A question needs clarification if: the \
scope is ambiguous, there's no clear decision criteria for a recommendation, \
the intended audience/use-case is unclear, or key constraints (budget, \
timeframe, etc.) are missing and would materially change the answer.

Do NOT ask clarifying questions if the question is already reasonably clear \
— err toward proceeding rather than over-asking. Ask at most 3 questions."""


def check_clarification(question: str) -> dict:
    """Returns {"needs_clarification": bool, "questions": [...]}"""
    response = client.messages.create(
        model=config.CLARIFIER_MODEL,
        max_tokens=500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Research question: {question}"}],
        tools=[CLARIFIER_TOOL],
        tool_choice={"type": "tool", "name": "emit_clarification_check"},
    )
    for block in response.content:
        if block.type == "tool_use":
            return block.input
    return {"needs_clarification": False, "questions": []}
