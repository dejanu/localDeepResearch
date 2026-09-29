"""Top-level orchestration: clarify -> plan -> run sub-agents (parallel) -> synthesize."""
import asyncio

from app.clarifier import check_clarification
from app.planner import create_plan
from app.subagent import run_subagent
from app.synthesizer import synthesize


async def run_research(question: str, clarification_answers: dict | None = None) -> dict:
    # If the caller already supplied clarification answers, fold them into
    # the question and skip re-checking (avoids an infinite clarification loop).
    if clarification_answers:
        answers_text = "\n".join(f"- {q}: {a}" for q, a in clarification_answers.items())
        question = f"{question}\n\nAdditional context from user:\n{answers_text}"
    else:
        clarification = check_clarification(question)
        if clarification.get("needs_clarification"):
            return {"status": "needs_clarification", "questions": clarification["questions"]}

    plan = create_plan(question)
    if not plan:
        return {
            "status": "complete",
            "result": {
                "recommendation": "Unable to generate a research plan for this question.",
                "confidence": "low",
                "reasoning_summary": "",
                "key_evidence": [],
            },
        }

    # Run all sub-agents concurrently. run_subagent is sync (uses the
    # blocking Anthropic client), so dispatch each to a thread.
    loop = asyncio.get_event_loop()
    tasks = [
        loop.run_in_executor(None, run_subagent, sq["question"], sq["sources"])
        for sq in plan
    ]
    subagent_results = await asyncio.gather(*tasks)

    result = synthesize(question, subagent_results)
    return {"status": "complete", "result": result}
