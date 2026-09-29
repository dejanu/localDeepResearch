"""CLI client: talks to the /research endpoint, handles the clarification round-trip."""
import json
import os
import click
import requests

API_URL = os.environ.get("RESEARCH_API_URL", "http://localhost:8000")


def _print_recommendation(result: dict):
    click.secho("\n=== RECOMMENDATION ===", fg="green", bold=True)
    click.echo(result.get("recommendation", ""))

    click.secho(f"\nConfidence: {result.get('confidence', 'unknown')}", fg="yellow")

    if result.get("reasoning_summary"):
        click.secho("\n--- Reasoning ---", bold=True)
        click.echo(result["reasoning_summary"])

    if result.get("key_evidence"):
        click.secho("\n--- Key Evidence ---", bold=True)
        for e in result["key_evidence"]:
            click.echo(f"  • [{e.get('source_type')}] {e.get('claim')}")
            click.echo(f"      source: {e.get('source')}")
            if e.get("excerpt"):
                click.echo(f"      excerpt: \"{e['excerpt']}\"")

    if result.get("conflicting_evidence"):
        click.secho("\n--- Conflicts Found ---", bold=True, fg="red")
        for c in result["conflicting_evidence"]:
            click.echo(f"  • {c.get('issue')}")
            click.echo(f"      web says: {c.get('web_says')}")
            click.echo(f"      docs say: {c.get('docs_say')}")

    if result.get("risks_or_caveats"):
        click.secho("\n--- Risks / Caveats ---", bold=True)
        for r in result["risks_or_caveats"]:
            click.echo(f"  • {r}")

    if result.get("alternatives_considered"):
        click.secho("\n--- Alternatives Considered ---", bold=True)
        for a in result["alternatives_considered"]:
            click.echo(f"  • {a}")

    if result.get("open_questions"):
        click.secho("\n--- Open Questions ---", bold=True)
        for q in result["open_questions"]:
            click.echo(f"  • {q}")


@click.command()
@click.option("--question", "-q", help="Research question. If omitted, you'll be prompted.")
@click.option("--save", type=click.Path(), help="Save the final result as JSON to this path.")
def main(question, save):
    if not question:
        question = click.prompt("Research question")

    clarification_answers = None

    while True:
        payload = {"question": question}
        if clarification_answers:
            payload["clarification_answers"] = clarification_answers

        resp = requests.post(f"{API_URL}/research", json=payload, timeout=300)
        resp.raise_for_status()
        data = resp.json()

        if data["status"] == "needs_clarification":
            click.secho("\nA few clarifying questions before I start:", fg="cyan", bold=True)
            clarification_answers = {}
            for q in data["questions"]:
                answer = click.prompt(f"  {q}")
                clarification_answers[q] = answer
            continue  # resubmit with answers

        if data["status"] == "complete":
            _print_recommendation(data["result"])
            if save:
                with open(save, "w") as f:
                    json.dump(data["result"], f, indent=2)
                click.secho(f"\nSaved to {save}", fg="green")
            break

        click.secho(f"Unexpected response: {data}", fg="red")
        break


if __name__ == "__main__":
    main()
