# lmagent (BigSmall)

Delegate simple but token-heavy work from a big cloud model (Claude Code) to small local models served by
LM Studio: Qwen, Gemma, Nemotron and similar. The local model reads the large input; the caller only
gets back a compact result.

## What it does

- Talks to LM Studio over its REST API (`localhost:1234`), including model load/unload (`/api/v1/models`).
- Routes tasks to model roles (`code`, `text`, `bulk`). Only one ~20 GB model fits in VRAM at a time, so
  by default it reuses the loaded model and switches to the role model only for big inputs.
- Splits big inputs into chunks that fit the loaded context, runs chunks in parallel, and merges partial
  results (map-reduce).
- Ships task templates: `ask`, `summarize`, `translate`, `extract`, `classify`, `rewrite`, `explain_diff`,
  `generate`.
- Writes long results to `.lmagent/out/` and can rewrite files in place.
- Logs every run to `~/.lmagent/log.jsonl` so you can see how many tokens were kept off the cloud model.

## Install

```bash
pip install -e .
```

Requires LM Studio with the local server running on `localhost:1234`. Everything, including model
load/unload, goes through its REST API; the `lms` CLI is not used.

## CLI

```bash
lmagent models                      # models and load state
lmagent tasks                       # task templates
lmagent load code                   # load by role or model id
lmagent run summarize -f logs/ -p focus=errors
lmagent run translate -f docs/intro.md -p to=English -o docs/intro.en.md
lmagent run extract -i "all HTTP endpoints with method and path" -f src/
lmagent run classify -i "Label as: test, config, source" -f src/
lmagent run rewrite -i "Convert var to const/let" -f "src/**/*.js" --in-place
lmagent run explain_diff --stdin < changes.diff
lmagent run ask -i "Which functions are never called?" -f src/
lmagent stats                       # offloaded token totals
```

Flags: `-m MODEL` forces a model, `--strict` forces the role model even if another one is loaded,
`-p key=value` fills template params, `-o FILE` writes the result, `--json` prints the full result object.

## Semantic search

```bash
lmagent index                         # build/update .lmagent/index/ for the current directory
lmagent search "where are retries handled" -k 5
lmagent search "database config" --files
```

Chunks of ~400 tokens are embedded with the local `embed` model (nomic, with its `search_document:` /
`search_query:` prefixes). Only new or changed files are re-embedded. The MCP tool `lm_search` refreshes the
index before every query, so there is no separate indexing step from Claude Code.

## MCP server for Claude Code

Register once (user scope, works in every project):

```bash
claude mcp add --scope user lmagent -- python "E:/Projects/lm agent/mcp_server.py"
```

Tools exposed: `lm_delegate`, `lm_summarize_files`, `lm_batch`, `lm_search`, `lm_index`, `lm_models`, `lm_tasks`.

A `local-worker` subagent (`~/.claude/agents/local-worker.md`, runs on Haiku) wraps these tools: the main
Claude session delegates a job to it, it dispatches to the local models and returns a compact answer, so the
big content never enters the main context.
Results longer than `output.inline_limit` characters are written to a file and only the path plus the
head of the text comes back, so the cloud context stays small.

## Config

Layers, later wins: `lmagent/default_config.yaml` < `~/.lmagent/config.yaml` < `./lmagent.yaml` <
`LMAGENT_CONFIG` / `--config`. Key options:

| Key | Meaning |
|---|---|
| `models.code/text/bulk/embed` | model id per role |
| `load.policy` | `smart` (default) switches to the role model only for inputs of at least `load.switch_min_tokens`; `prefer_loaded` never switches; `strict` always loads the role model |
| `load.context_length`, `load.ttl`, `load.parallel` | passed to `lms load` |
| `server.retries`, `server.retry_delay` | retries on timeouts, connection errors, 5xx and unloaded-model errors (the model is reloaded automatically) |
| `generation.max_tokens`, `generation.reserve_tokens` | output cap per call; output room reserved per worker when planning chunks |
| `generation.thinking` | `false` sends `reasoning_effort: none`, which is what actually disables reasoning in LM Studio (Qwen and Nemotron verified) |
| `chunking.chunk_tokens`, `chunking.max_parallel` | chunk size and concurrency |
| `chunking.token_safety` | model tokenizers count ~1.1x (code) to ~1.5x (logs) more than tiktoken; the context budget is divided by this. If a chunk still overflows, it is split in half and retried |
| `output.inline_limit` | chars returned inline before spilling to a file |

## Offload report

```bash
lmagent stats                    # per day, all time
lmagent stats --days 7 --by task # per task, last week; also --by model, --by cwd, --json
```

Shows runs, calls, input/output tokens handled locally, wall time and model load time.

## Layout

```
lmagent/
  client.py     LM Studio REST client (discovery, load/unload, chat, embeddings, retries)
  config.py     layered YAML config
  chunker.py    token counting, splitting, file discovery
  index.py      incremental embedding index and semantic search
  tasks/        task templates (prompt, role, mode)
  runner.py     orchestration: routing, chunking, parallel map-reduce, output, log
  cli.py        command line
mcp_server.py   MCP stdio server for Claude Code
```
