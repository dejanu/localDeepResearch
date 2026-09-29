"""
Sub-agent: a single, tool-scoped Claude conversation that researches one
sub-question until it decides it has enough, or hits the iteration cap.

Not a separate service — just an isolated `messages` array + a bounded loop.
"""
import anthropic

from app.config import config
from app.tools import execute_local_tool_call, WEB_SEARCH_TOOL, SEARCH_LOCAL_DOCS_TOOL

client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

SUBAGENT_SYSTEM_PROMPT = """You are a focused research sub-agent. You will be \
given one specific sub-question and a set of tools. Use the tools to gather \
evidence, iterating as needed. When you have enough information, STOP calling \
tools and respond with a final plain-text summary formatted as a list of \
findings, one per line, in this exact format:

CLAIM: <the finding> | SOURCE_TYPE: <web or local_doc> | SOURCE: <url or filename> | EXCERPT: <short supporting quote>

If you cannot find relevant information after a reasonable effort, say so \
explicitly rather than fabricating findings."""


def _tools_for_sources(sources: list[str]) -> list[dict]:
    tools = []
    if "web" in sources:
        tools.append(WEB_SEARCH_TOOL)
    if "local_docs" in sources:
        tools.append(SEARCH_LOCAL_DOCS_TOOL)
    return tools


def _execute_tool_use_block(block) -> dict:
    """Execute one tool_use block, returning a tool_result content block.
    web_search is executed server-side by Anthropic already, so we only
    need to handle our custom tools here."""
    if block.name == "search_local_docs":
        result_text = execute_local_tool_call(block.name, block.input)
        return {
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": result_text,
        }
    # web_search results arrive already resolved in the same response;
    # this branch shouldn't be reached for it, but guard anyway.
    return {
        "type": "tool_result",
        "tool_use_id": block.id,
        "content": "",
    }


def parse_findings(text: str) -> list[dict]:
    findings = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("CLAIM:"):
            continue
        parts = {}
        for segment in line.split(" | "):
            if ":" in segment:
                key, _, value = segment.partition(":")
                parts[key.strip().upper()] = value.strip()
        if "CLAIM" in parts:
            findings.append({
                "claim": parts.get("CLAIM", ""),
                "source_type": parts.get("SOURCE_TYPE", "unknown"),
                "source": parts.get("SOURCE", ""),
                "excerpt": parts.get("EXCERPT", ""),
            })
    if not findings and text.strip():
        # Fallback: no structured lines found, keep the raw text as one finding
        findings.append({
            "claim": text.strip(),
            "source_type": "unknown",
            "source": "",
            "excerpt": "",
        })
    return findings


def run_subagent(sub_question: str, sources: list[str]) -> dict:
    """Run one bounded research loop for a single sub-question.

    sources: subset of ["web", "local_docs"]
    Returns: {"sub_question": ..., "findings": [...], "iterations_used": N}
    """
    tools = _tools_for_sources(sources)
    messages = [{"role": "user", "content": f"Research question: {sub_question}"}]

    iterations_used = 0
    for i in range(config.MAX_SUBAGENT_ITERATIONS):
        iterations_used += 1
        response = client.messages.create(
            model=config.SUBAGENT_MODEL,
            max_tokens=2000,
            system=SUBAGENT_SYSTEM_PROMPT,
            messages=messages,
            tools=tools,
        )

        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]

        if not tool_use_blocks:
            # Claude is done — extract final text
            text = "".join(b.text for b in response.content if b.type == "text")
            return {
                "sub_question": sub_question,
                "findings": parse_findings(text),
                "iterations_used": iterations_used,
            }

        messages.append({"role": "assistant", "content": response.content})

        # Only need to execute our own custom tools; web_search tool_use
        # blocks are auto-resolved server-side and appear alongside their
        # results already, so we just need to supply results for anything
        # that still needs one (search_local_docs).
        tool_results = []
        for block in tool_use_blocks:
            if block.name == "search_local_docs":
                tool_results.append(_execute_tool_use_block(block))

        if tool_results:
            messages.append({"role": "user", "content": tool_results})
        else:
            # Nothing left for us to respond to (e.g. only web_search was
            # used and it's already resolved) — request the wrap-up directly.
            messages.append({
                "role": "user",
                "content": "Please provide your final findings now in the required format.",
            })

    # Hit iteration cap without a clean stop — ask once more for a summary.
    response = client.messages.create(
        model=config.SUBAGENT_MODEL,
        max_tokens=2000,
        system=SUBAGENT_SYSTEM_PROMPT,
        messages=messages + [{
            "role": "user",
            "content": "You've reached the research limit. Summarize your findings now in the required format.",
        }],
        tools=[],
    )
    text = "".join(b.text for b in response.content if b.type == "text")
    return {
        "sub_question": sub_question,
        "findings": parse_findings(text),
        "iterations_used": iterations_used,
    }
