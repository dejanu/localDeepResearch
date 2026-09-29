"""Tool schema definitions passed to the Anthropic API."""
from app.local_docs import store

WEB_SEARCH_TOOL = {"type": "web_search_20250305", "name": "web_search"}

SEARCH_LOCAL_DOCS_TOOL = {
    "name": "search_local_docs",
    "description": (
        "Semantic search over the local document corpus (PDFs, Markdown, "
        "JSON, and text files). Returns the most relevant chunks with their "
        "source filename."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "top_k": {"type": "integer", "description": "Number of results to return", "default": 5},
        },
        "required": ["query"],
    },
}

# Forced structured output for the synthesizer (used as a tool_use call, not
# a real "tool" — see synthesizer.py for how tool_choice pins this).
RECOMMENDATION_SCHEMA_TOOL = {
    "name": "emit_recommendation",
    "description": "Emit the final structured research recommendation.",
    "input_schema": {
        "type": "object",
        "properties": {
            "recommendation": {"type": "string"},
            "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
            "reasoning_summary": {"type": "string"},
            "key_evidence": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "claim": {"type": "string"},
                        "source_type": {"type": "string", "enum": ["web", "local_doc"]},
                        "source": {"type": "string"},
                        "excerpt": {"type": "string"},
                    },
                    "required": ["claim", "source_type", "source"],
                },
            },
            "conflicting_evidence": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "issue": {"type": "string"},
                        "web_says": {"type": "string"},
                        "docs_say": {"type": "string"},
                    },
                },
            },
            "risks_or_caveats": {"type": "array", "items": {"type": "string"}},
            "alternatives_considered": {"type": "array", "items": {"type": "string"}},
            "open_questions": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["recommendation", "confidence", "reasoning_summary", "key_evidence"],
    },
}


def execute_local_tool_call(name: str, tool_input: dict) -> str:
    """Execute a non-web custom tool call and return its result as a string."""
    if name == "search_local_docs":
        results = store.search(tool_input["query"], tool_input.get("top_k"))
        return str(results)
    raise ValueError(f"Unknown tool: {name}")
