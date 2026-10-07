# Common uses of the lmagent pipeline

The pattern is always the same: the local model reads the big input, the caller (you in a shell, or Claude
Code through MCP) gets a compact result. Every recipe below shows the CLI form and the MCP form. In Claude
Code the MCP calls are made by Claude itself, so the "MCP" lines are what you would say to Claude, plus the
tool call it should make. `cwd` is always the absolute project directory.

Rule of thumb for when to delegate: a file or log over ~5k tokens, many files, mechanical bulk edits,
translation, "where is X handled". Not for small reads or anything that needs judgement on the final code.

## 1. What is in this huge log?

```bash
lmagent run summarize -f logs/app.log -p focus=errors
lmagent run summarize -f logs/ -i "Group errors by component, give counts and the first timestamp of each."
```

MCP: *"Summarize logs/app.log, focus on errors"* ->
`lm_summarize_files(files=["logs/app.log"], focus="errors", cwd=...)`.

A 45k-token log is split into chunks, processed in parallel and merged (map-reduce): ~40 s, the caller
receives 20-25 bullets. Logs count ~1.45x more tokens in the model than tiktoken says; the planner
already allows for that.

## 2. Where is X handled?

```bash
lmagent search "where is the retry logic for failed requests" --kind code
lmagent search "websocket reconnect" --kind code --files      # one hit per file
```

MCP: `lm_search(query="where is the retry logic", kind="code", cwd=...)` -> file, line range, score,
snippet per hit. Use `kind="code"` for these questions: README and docs otherwise outrank the code.
Then hand the hit files to `lm_delegate`.

## 3. How does X work? (answer, not hits)

```bash
lmagent search "retry logic" --kind code --ask "Which errors trigger a retry and with what backoff?"
```

MCP: `lm_search(query="retry logic", kind="code", ask="Which errors trigger a retry...", cwd=...)` ->
only the answer with `file:start-end` citations; the snippets stay local. About 5k tokens offloaded,
5-6 s. Spot-check one citation with grep before relying on it.

## 4. Pull structured data out of code or docs

```bash
lmagent run extract -i "every HTTP endpoint as {method, path, handler}" -f src/api/
lmagent run extract -i "every MCP tool as {name, file, required_params}" -f src/tools/*.ts -o tools.json
```

MCP: `lm_delegate(task="extract", instruction="every HTTP endpoint as {method, path, handler}",
files=["src/api/"], cwd=...)` -> a JSON array. In the live trial 28/28 tool names came back correct.

## 5. Label many files

```bash
lmagent run classify -i "Label as: test, config, source, docs, generated" -f src/
```

MCP: `lm_delegate(task="classify", instruction="Label as: test, config, source", files=["src/"], cwd=...)`
-> one `{file, label, confidence, reason}` per file (JSON schema enforced).

## 6. Mechanical bulk edit, safely

```bash
# 1. dry run: results go to .lmagent/out/<path>, the sources are untouched
lmagent run rewrite -i "Add a one-line JSDoc above every exported function that lacks one. Change nothing else." -f "src/**/*.ts"
# 2. inspect
diff -r src .lmagent/out/src
# 3. apply: originals copied to .lmagent/backup/<timestamp>/ first
lmagent run rewrite -i "..." -f "src/**/*.ts" --in-place
```

MCP: `lm_delegate(task="rewrite", instruction="...", files=["src/**/*.ts"], cwd=...)` first, then the
same with `in_place=true`. Each result line carries `(+added -removed)`; `.py`, `.json` and `.yaml`
outputs are parsed before writing and an invalid or empty result is reported as `NOT written`, the
original stays. Line endings are preserved. Only use `in_place` on files under version control.

## 7. Translate documentation

```bash
lmagent run translate -f docs/intro.md -p to=English -o docs/intro.en.md
lmagent run translate -f docs/ -p to=Ukrainian                       # one output per file in .lmagent/out/
```

MCP: `lm_delegate(task="translate", files=["docs/intro.md"], params={"to": "English"},
output_file="docs/intro.en.md", cwd=...)`. Markdown, code blocks and URLs are kept. For higher
quality at 3x the time use `model="google/gemma-4-31b"` (see `examples/translation.yaml`).

## 8. Explain a big diff before review

```bash
git diff main...HEAD | lmagent run explain_diff --stdin
git show HEAD~3..HEAD > changes.diff && lmagent run explain_diff -f changes.diff
```

MCP: `lm_delegate(task="explain_diff", text="<diff>", cwd=...)` or with `files=["changes.diff"]` ->
what changed, why, risks to double-check.

## 9. Generate boilerplate from reference files

```bash
lmagent run generate -i "pytest tests for every public function, one test per branch" -f src/chunker.py -o tests/test_chunker.py
lmagent run generate -i "a README section documenting these CLI flags" -f src/cli.py
```

MCP: `lm_delegate(task="generate", instruction="pytest tests for ...", files=["src/chunker.py"],
output_file="tests/test_chunker.py", cwd=...)`. Review the output: generation needs judgement, the
local model only saves the typing.

## 10. Several independent jobs at once

MCP: `lm_batch(task="summarize", items=[{"files": ["logs/a.log"]}, {"files": ["logs/b.log"]},
{"files": ["logs/c.log"], "params": {"focus": "timeouts"}}], cwd=...)` -> one compact result per item,
in order, run in parallel. Progress notifications say `batch item n/N`.

## 11. Keep the main context small: the local-worker subagent

In Claude Code, ask for the subagent instead of the tools directly:

> Use the local-worker subagent on E:\Projects\MyApp: find where the payment webhook is verified and
> summarize how the signature check works, with file and line citations.

The subagent (`~/.claude/agents/local-worker.md`, Haiku) only has the `lm_*` tools and `Glob`, never reads
files itself, and returns a report under 300 words. Always name the project directory in the prompt so it
can pass `cwd`. Line numbers in its citations were off by a few lines in the trial: verify before quoting.

## 12. Project intake from Worksection (the local model drives an MCP server)

```bash
lmagent intake https://<acct>.worksection.com/project/348940/22729222/      # one task
lmagent intake https://<acct>.worksection.com/project/348940/               # whole project, task by task + merge
```

MCP: say *"читай агентом <link>"* -> `lm_intake(link, cwd=<project folder>)`. LM Studio runs its own agent
loop (`POST /api/v1/chat` with `integrations`): the local model calls the Worksection MCP tools itself
(task, discussion, attachments, `get_file_content` for pdf/docx/xlsx/pptx) and writes `intake/digest.md`
with fixed sections (client, scope, deliverables, deadlines, constraints, attachments, discussion, open
questions, facts for the estimate). Every tool output is kept in `intake/raw/NN_<tool>.json`, the call log
in `intake/run.json`. Claude reads only the digest.

Requirements: the Worksection MCP server running (`uv run python -m worksection_mcp`, port 8000), listed in
`~/.lmstudio/mcp.json` under the label from `intake.mcp`; LM Studio Server Settings: Require Authentication
ON with an API token in `server.api_key` (or `LMSTUDIO_API_KEY`) and "Allow calling servers from mcp.json"
ON. `ephemeral_mcp` does not work for local addresses. Tool whitelist: `intake.allowed_tools`.

## 13. How much stayed local?

```bash
lmagent stats --days 7 --by task
lmagent stats --by cwd
```

Totals of prompt/completion tokens handled by the local model, time and model-load seconds, from
`~/.lmagent/log.jsonl`. Every MCP result also carries `offloaded_tokens` and `elapsed_s`.

## Troubleshooting

- *LM Studio server is not reachable* -> start the server (LM Studio -> Developer -> Start Server, or
  `lms server start`), or fix `server.base_url`. Nothing is retried on a dead server.
- A result ends with `...[truncated, full result in ...]` -> the full text is in `.lmagent/out/`; pass
  `output_file` to choose the path.
- `rewrite` says `NOT written (SyntaxError ...)` -> the model broke the file; tighten the instruction
  ("change nothing else", "output the complete file") or split the job per file.
- Search hits are all markdown -> add `--kind code`; vendored or generated folders flood the index ->
  `index.exclude` with `dir/` patterns (see `monorepo.yaml`).
- The MCP server does not know a new parameter -> Claude Code starts the server once per session; open a
  new session.
