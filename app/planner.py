"""Planner: decomposes the (clarified) research question into sub-questions."""
import anthropic

from app.config import config

client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

PLANNER_TOOL = {
    "name": "emit_plan",
    "description": "Emit the research plan: a set of sub-questions with assigned sources.",
    "input_schema": {
        "type": "object",
        "properties": {
            "sub_questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                        "sources": {
                            "type": "array",
                            "items": {"type": "string", "enum": ["web", "local_docs"]},
                            "description": "Which source(s) are relevant to this sub-question",
                        },
                    },
                    "required": ["question", "sources"],
                },
            }
        },
        "required": ["sub_questions"],
    },
}

SYSTEM_PROMPT = """You are a research planner. Break the given research \
question into 2-5 focused sub-questions that together cover what's needed \
to answer it well. For each sub-question, decide which source(s) are \
relevant:
- "web": for current events, market data, external facts, general knowledge
- "local_docs": for anything that would live in the user's own documents \
(internal notes, specs, prior analysis, proprietary data)
- both, if genuinely relevant to cross-check

Keep sub-questions specific and independently researchable."""


def create_plan(question: str) -> list[dict]:
    response = client.messages.create(
        model=config.PLANNER_MODEL,
        max_tokens=1000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Research question: {question}"}],
        tools=[PLANNER_TOOL],
        tool_choice={"type": "tool", "name": "emit_plan"},
    )
    for block in response.content:
        if block.type == "tool_use":
            return block.input["sub_questions"]
    return []
