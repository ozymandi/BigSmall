from __future__ import annotations

import argparse
import json
import sys

from .client import LMStudioClient, LMStudioError
from .config import load_config
from .runner import Runner
from .tasks import TASKS


def _parse_params(pairs: list[str]) -> dict:
    out = {}
    for p in pairs or []:
        if "=" not in p:
            raise SystemExit(f"bad param '{p}', expected key=value")
        k, v = p.split("=", 1)
        out[k.strip()] = v
    return out


def cmd_models(args, cfg):
    client = LMStudioClient(cfg["server"]["base_url"], cfg["server"]["timeout"])
    roles = {v: k for k, v in cfg["models"].items()}
    for m in client.models():
        state = "LOADED" if m.get("state") == "loaded" else "-"
        ctx = m.get("loaded_context_length") or ""
        role = roles.get(m["id"], "")
        print(f"{state:6} {m['id']:45} {m.get('type', ''):4} {str(ctx):>7} {role}")


def cmd_tasks(args, cfg):
    for name, spec in TASKS.items():
        print(f"{name:13} [{spec.role:4}] {spec.mode:10} {spec.description}")
        if spec.defaults:
            print(f"{'':13} params: " + ", ".join(f"{k}={v}" for k, v in spec.defaults.items()))


def cmd_load(args, cfg):
    client = LMStudioClient(cfg["server"]["base_url"], cfg["server"]["timeout"])
    model = cfg["models"].get(args.model, args.model)
    ld = cfg["load"]
    did = client.load(model, args.context or ld["context_length"], ld["ttl"], ld["parallel"], ld["unload_others"])
    print(f"{'loaded' if did else 'already loaded'}: {model}")


def cmd_unload(args, cfg):
    client = LMStudioClient(cfg["server"]["base_url"], cfg["server"]["timeout"])
    targets = [m["id"] for m in client.loaded_llms()] if args.all or not args.model else [args.model]
    for t in targets:
        client.unload(t)
        print(f"unloaded: {t}")


def cmd_run(args, cfg):
    text = args.text or ""
    if args.stdin:
        text = sys.stdin.read()
    runner = Runner(cfg)
    r = runner.run(
        args.task, instruction=args.instruction or "", files=args.files or [], text=text,
        model=args.model, strict=args.strict, params=_parse_params(args.param),
        output=args.output, in_place=args.in_place,
    )
    if args.json:
        print(json.dumps(r.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(r.text)
        if r.output_path:
            print(f"\n[saved to {r.output_path}]", file=sys.stderr)
    print(r.stats_line(), file=sys.stderr)
    for n in r.notes:
        print(f"  note: {n}", file=sys.stderr)


def _index(cfg, root):
    from .index import Index
    from pathlib import Path
    root = Path(root) if root else Path(cfg["_cwd"])
    return Index(cfg, LMStudioClient(cfg["server"]["base_url"], cfg["server"]["timeout"]), root)


def cmd_index(args, cfg):
    s = _index(cfg, args.root).update(args.paths or None)
    print(f"indexed {s['indexed']} file(s) ({s['chunks']} chunks), unchanged {s['unchanged']}, "
          f"removed {s['removed']}, skipped {s['skipped']}; total {s['total_files']} files / {s['total_chunks']} chunks")


def cmd_search(args, cfg):
    idx = _index(cfg, args.root)
    if not args.no_update:
        idx.update(None)
    hits = idx.search(args.query, k=args.k, files_only=args.files)
    if args.json:
        print(json.dumps(hits, ensure_ascii=False, indent=2))
        return
    for h in hits:
        print(f"{h['score']:.3f}  {h['file']}:{h['start_line']}-{h['end_line']}")
        if not args.files:
            snippet = h["text"].strip().splitlines()
            for line in snippet[:args.lines]:
                print(f"        {line[:160]}")
            print()


def cmd_stats(args, cfg):
    s = Runner(cfg).stats(days=args.days, by=args.by)
    if args.json:
        print(json.dumps(s, ensure_ascii=False, indent=2))
        return
    label = {"day": "date", "task": "task", "model": "model", "cwd": "project"}[args.by]
    rows = list(s["groups"].items()) + [("TOTAL", s["total"])]
    width = max(len(label), *(len(k) for k, _ in rows))
    print(f"{label:{width}}  {'runs':>5} {'calls':>5} {'in tok':>10} {'out tok':>9} {'time':>8} {'load':>6}")
    for key, g in rows:
        if key == "TOTAL":
            print("-" * (width + 50))
        print(f"{key:{width}}  {g['runs']:>5} {g['calls']:>5} {g['prompt_tokens']:>10,} "
              f"{g['completion_tokens']:>9,} {g['elapsed']:>7.0f}s {g['load_s']:>5.0f}s")
    period = f"last {args.days} days" if args.days else "all time"
    print(f"\nOffloaded from the cloud model ({period}): "
          f"{s['total']['prompt_tokens'] + s['total']['completion_tokens']:,} tokens")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="lmagent", description="Delegate token-heavy tasks to local LM Studio models.")
    p.add_argument("--config", help="path to a config yaml (overrides ~/.lmagent/config.yaml)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("models", help="list models and their load state").set_defaults(fn=cmd_models)
    sub.add_parser("tasks", help="list task templates").set_defaults(fn=cmd_tasks)

    s = sub.add_parser("load", help="load a model (by id or role: code/text/bulk)")
    s.add_argument("model")
    s.add_argument("--context", type=int)
    s.set_defaults(fn=cmd_load)

    s = sub.add_parser("unload", help="unload a model (default: all loaded LLMs)")
    s.add_argument("model", nargs="?")
    s.add_argument("--all", action="store_true")
    s.set_defaults(fn=cmd_unload)

    s = sub.add_parser("run", help="run a task")
    s.add_argument("task", choices=list(TASKS))
    s.add_argument("-i", "--instruction", help="what to do")
    s.add_argument("-f", "--files", nargs="+", help="files, directories or globs")
    s.add_argument("-t", "--text", help="inline text input")
    s.add_argument("--stdin", action="store_true", help="read text input from stdin")
    s.add_argument("-m", "--model", help="model id override")
    s.add_argument("--strict", action="store_true", help="always use the role model, even if another is loaded")
    s.add_argument("-p", "--param", action="append", help="template param key=value (e.g. to=English, focus=errors)")
    s.add_argument("-o", "--output", help="write the result to this file")
    s.add_argument("--in-place", action="store_true", help="rewrite: overwrite source files instead of writing to .lmagent/out")
    s.add_argument("--json", action="store_true", help="print the full result as JSON")
    s.set_defaults(fn=cmd_run)

    s = sub.add_parser("index", help="build or update the embedding index of a directory")
    s.add_argument("paths", nargs="*", help="files/dirs/globs to index (default: whole root)")
    s.add_argument("--root", help="index root (default: current directory)")
    s.set_defaults(fn=cmd_index)

    s = sub.add_parser("search", help="semantic search over the embedding index")
    s.add_argument("query")
    s.add_argument("-k", type=int, default=8, help="number of hits")
    s.add_argument("--files", action="store_true", help="one hit per file")
    s.add_argument("--lines", type=int, default=6, help="snippet lines to show per hit")
    s.add_argument("--no-update", action="store_true", help="do not refresh the index before searching")
    s.add_argument("--root", help="index root (default: current directory)")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_search)

    s = sub.add_parser("stats", help="offloaded token report from the run log")
    s.add_argument("--days", type=int, help="only the last N days (default: all)")
    s.add_argument("--by", choices=["day", "task", "model", "cwd"], default="day")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_stats)
    return p


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    try:
        args.fn(args, cfg)
    except (LMStudioError, ValueError, KeyError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
