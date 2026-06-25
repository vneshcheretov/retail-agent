"""Command-line chat interface for the retail data agent.

Run:
    python -m src.cli            # default user
    python -m src.cli --user vadim

Commands inside the chat:
    /help                 show commands
    /persona [name]       list personas, or switch the active one
    /user <name>          switch user (memory is per user)
    /memory               show what the agent has learned about you
    /forget               clear your learned memory
    /save                 add the last answered question to the golden bucket
    /quit                 exit
"""

from __future__ import annotations

import argparse

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from src.graph import build_graph
from src.services.golden_bucket import GoldenBucket, Trio
from src.services.memory import build_store, clear_memory, load_memory, reflect
from src.services.observability import RunMetrics, RunObserver
from src.services.personas import list_personas, set_active_persona
from src.state import AgentState

console = Console()

# Re-learn the user's memory after this many ANALYSIS turns (raise to batch and
# save LLM calls; 1 = update after every analytical question).
REFLECT_EVERY = 1


def render_rows(rows: list[dict]) -> Table | None:
    """Render up to 10 result rows as a table (already PII-masked)."""
    if not rows:
        return None
    table = Table(show_header=True, header_style="bold cyan")
    for column in rows[0].keys():
        table.add_column(str(column))
    for row in rows[:10]:
        table.add_row(*(str(v) for v in row.values()))
    return table


def render_summary(m: RunMetrics) -> Panel:
    """Render the per-run observability summary."""
    nodes = "  ".join(f"{name}={ms}ms" for name, ms in m.node_ms.items())
    text = (
        f"run={m.run_id}  status={m.status}  total={m.total_ms}ms\n"
        f"llm_calls={m.llm_calls}  tokens(in/out)={m.input_tokens}/{m.output_tokens}\n"
        f"sql_attempts={m.sql_attempts}  sql_errors={m.sql_errors}  "
        f"empty={m.empty_results}  pii_redactions={m.pii_redactions}\n"
        f"nodes: {nodes}"
    )
    return Panel(text, title="run summary", border_style="dim", expand=False)


def print_help() -> None:
    console.print(
        Panel(
            "/persona [name]   list or switch persona\n"
            "/user <name>   switch user\n"
            "/memory   show what the agent learned about you\n"
            "/forget   clear your learned memory\n"
            "/save   add the last question to the golden bucket\n"
            "/quit   exit",
            title="commands",
            border_style="dim",
        )
    )


def handle_command(cmd: str, ctx: dict) -> bool:
    """Run a slash command. Returns False to exit the chat."""
    parts = cmd.split()
    name = parts[0]

    if name in ("/quit", "/exit"):
        return False
    if name == "/help":
        print_help()
    elif name == "/user":
        ctx["user"] = parts[1] if len(parts) > 1 else ctx["user"]
        console.print(f"[green]Switched to user '{ctx['user']}'.[/green]")
    elif name == "/memory":
        profile = load_memory(ctx["user"])
        style = (profile.get("answer_style") or "").strip()
        interests = profile.get("interests") or []
        if not style and not interests:
            console.print("Nothing learned yet — keep chatting.")
        else:
            lines = []
            if style:
                lines.append(f"Answer style: {style}")
            if interests:
                lines.append(f"Interests: {', '.join(interests)}")
            console.print(Panel("\n".join(lines), title=f"memory: {ctx['user']}", border_style="dim"))
    elif name == "/forget":
        clear_memory(ctx["store"], ctx["user"])
        console.print(f"[green]Cleared memory for '{ctx['user']}'.[/green]")
    elif name == "/persona":
        if len(parts) > 1:
            ok = set_active_persona(parts[1])
            console.print(
                f"[green]Persona set to '{parts[1]}'.[/green]" if ok
                else f"[red]Unknown persona '{parts[1]}'.[/red]"
            )
        else:
            for key, label in list_personas().items():
                console.print(f"  {key}: {label}")
    elif name == "/save":
        last = ctx.get("last")
        if last and last.get("masked_rows"):
            GoldenBucket().add_trio(
                Trio(question=last["question"], sql=last["sql"], report=last["answer"])
            )
            console.print("[green]Saved to the golden bucket.[/green]")
        else:
            console.print("Nothing to save yet.")
    else:
        console.print("Unknown command. Type /help.")
    return True


def answer_question(app, ctx: dict, question: str) -> None:
    """Run one turn through the graph and print the result and summary."""
    obs = RunObserver(verbose=ctx.get("verbose", False))
    state: AgentState = {
        "question": question,
        "user": ctx["user"],
        "history": ctx["history"][-6:],
        "observer": obs,
        "attempts": 0,
    }
    try:
        result = app.invoke(state)
        intent = result.get("intent", "analysis" if result.get("allowed") else "blocked")
    except Exception as exc:  # noqa: BLE001 - keep the chat alive on any failure
        console.print(Panel(f"Something went wrong: {exc}", border_style="red"))
        obs.finish(status="error")
        return

    if intent == "blocked":
        answer = result.get("refusal", "Sorry, I can't help with that request.")
        console.print(Panel(Markdown(answer), title="blocked", border_style="yellow"))
    elif intent == "redirect":
        answer = result.get("answer", "Hi! Ask me about sales, products, customers, or orders.")
        console.print(Panel(Markdown(answer), title="assistant", border_style="cyan"))
    else:  # analysis
        answer = result.get("answer", "No answer.")
        console.print(Panel(Markdown(answer), title="answer", border_style="green"))
        table = render_rows(result.get("masked_rows", []))
        if table is not None:
            console.print(table)

    status = "ok" if intent == "analysis" else intent
    metrics = obs.finish(status=status)
    console.print(render_summary(metrics))

    ctx["history"].append({"role": "user", "content": question})
    ctx["history"].append({"role": "assistant", "content": answer})
    ctx["last"] = result
    return intent


def maybe_reflect(ctx: dict, force: bool = False) -> None:
    """Update the user's learned memory after enough analysis turns (or on exit)."""
    if not ctx["history"] or (not force and ctx["turns_since_reflect"] < REFLECT_EVERY):
        return
    try:
        reflect(ctx["store"], ctx["user"], ctx["history"])
        ctx["turns_since_reflect"] = 0
    except Exception as exc:  # noqa: BLE001 - memory is best-effort, never break the chat
        console.print(f"[dim](memory update skipped: {exc})[/dim]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Retail data chat agent")
    parser.add_argument("--user", default="default", help="user name (memory is per user)")
    parser.add_argument("--verbose", action="store_true", help="print each step (intent, SQL, trios, report prompt) in flow")
    args = parser.parse_args()

    console.print(
        Panel(
            "Retail Data Agent. Ask about sales, products, customers, or orders.\n"
            "Type /help for commands, /quit to exit.",
            border_style="cyan",
        )
    )
    store = build_store()  # LangGraph store for per-user memory (hydrated from disk)
    app = build_graph(store=store)

    # Load the schema/enum context. It is disk-cached, so this is instant on later
    # runs; only the very first run profiles BigQuery (~1 min) behind a spinner.
    from src.services.bigquery_client import context_cached, get_enums, get_schema

    def _load_context() -> None:
        try:
            get_schema()
            get_enums()
        except Exception:  # noqa: BLE001 - no creds/offline: fall back at query time
            pass

    if context_cached():
        _load_context()
    else:
        with console.status("First run: profiling the schema from BigQuery (cached next time, ~1 min)..."):
            _load_context()

    ctx: dict = {"user": args.user, "history": [], "last": None, "turns_since_reflect": 0, "verbose": args.verbose, "store": store}

    while True:
        try:
            user_input = console.input(f"\n[bold]({ctx['user']}) >[/bold] ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_input:
            continue
        if user_input.startswith("/"):
            if not handle_command(user_input, ctx):
                break
            continue
        intent = answer_question(app, ctx, user_input)
        if intent == "analysis":  # greetings/blocked carry no signal to learn from
            ctx["turns_since_reflect"] += 1
            maybe_reflect(ctx)

    maybe_reflect(ctx, force=True)  # capture the tail of the session
    console.print("\nGoodbye!")


if __name__ == "__main__":
    main()
