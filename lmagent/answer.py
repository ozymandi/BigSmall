from __future__ import annotations

from .chunker import count_tokens
from .index import Index
from .runner import Runner
from .tasks import get_task

ANSWER_INSTRUCTION = (
    "Answer the question using only the excerpts below. After each claim cite its source as "
    "`file:start-end`, exactly as in the excerpt header. If the excerpts do not contain the answer, "
    "say what is missing instead of guessing.\n\nQuestion: {question}"
)


def answer(runner: Runner, idx: Index, query: str, question: str = "", k: int = 12,
           kind: str = "all") -> dict:
    """Search, then let the local model answer from the top hits. Only the answer and the citations
    come back; the hit texts stay local. The excerpts are capped by the planner's chunk budget so the
    whole thing is one model call."""
    question = question or query
    hits = idx.search(query, k=k, kind=kind)
    spec = get_task("ask")
    notes: list[str] = []
    model = runner.resolve_model(spec, None, False, notes, 0)
    budget = runner._plan(model, spec, notes) - count_tokens(question) - 200
    parts: list[str] = []
    sources: list[dict] = []
    used = 0
    for h in hits:
        block = f"### {h['file']}:{h['start_line']}-{h['end_line']}\n{h['text'].rstrip()}\n"
        t = count_tokens(block)
        if parts and used + t > budget:
            break
        parts.append(block)
        used += t
        sources.append({key: v for key, v in h.items() if key != "text"})
    if not parts:
        return {"answer": "", "sources": [], "notes": notes + ["no hits"]}
    r = runner.run("ask", instruction=ANSWER_INSTRUCTION.format(question=question),
                   text="\n".join(parts), model=model)
    return {"answer": r.text, "sources": sources, "model": r.model,
            "offloaded_tokens": {"in": r.prompt_tokens, "out": r.completion_tokens},
            "elapsed_s": round(r.elapsed, 1), "notes": notes + r.notes}
