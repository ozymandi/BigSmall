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


def cmd_stats(args, cfg):
    s = Runner(cfg).stats()
    print(json.dumps(s, ensure_ascii=False, indent=2))


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

    sub.add_parser("stats", help="token usage totals from the log").set_defaults(fn=cmd_stats)
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
