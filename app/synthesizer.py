"""Synthesizer: reconciles all sub-agent findings into one structured recommendation."""
import json
import anthropic

from app.config import config
from app.tools import RECOMMENDATION_SCHEMA_TOOL

client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

SYSTEM_PROMPT = """You are a research synthesizer. You are given the \
original research question and a collection of findings gathered by \
research sub-agents (each tagged with source_type: "web" or "local_doc").

Your job:
1. Weigh all findings together and produce ONE clear recommendation/decision.
2. Explicitly surface any disagreements between web sources and local \
documents — do not silently prefer one over the other; report both sides \
in conflicting_evidence.
3. Assign a confidence level based on the quality/consistency of evidence.
4. List real risks/caveats and alternatives that were considered, even if \
ultimately not recommended.
5. If there are important gaps the sub-agents couldn't fill, list them as \
open_questions rather than guessing.

Be concrete and decision-useful, not vague."""


def synthesize(question: str, subagent_results: list[dict]) -> dict:
    findings_blob = json.dumps(subagent_results, indent=2)
    response = client.messages.create(
        model=config.SYNTHESIZER_MODEL,
        max_tokens=3000,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": (
                f"Original research question: {question}\n\n"
                f"Sub-agent findings:\n{findings_blob}"
            ),
        }],
        tools=[RECOMMENDATION_SCHEMA_TOOL],
        tool_choice={"type": "tool", "name": "emit_recommendation"},
    )
    for block in response.content:
        if block.type == "tool_use":
            return block.input
    return {
        "recommendation": "Synthesis failed to produce structured output.",
        "confidence": "low",
        "reasoning_summary": "",
        "key_evidence": [],
    }
