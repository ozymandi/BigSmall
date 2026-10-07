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

## Pipeline

### Entry points

```
A) Claude Code ──► lm_* MCP tool ──────────────────────────► lmagent core ──► LM Studio
B) Claude Code ──► Agent(local-worker, Haiku) ──► lm_* tool ──► lmagent core ──► LM Studio
C) shell       ──► lmagent run / search ───────────────────► lmagent core ──► LM Studio
```

- **A, direct tool call.** The main session calls `lm_delegate`, `lm_summarize_files`, `lm_batch` or
  `lm_search`. The local model reads the inputs; the main context receives only the compact result
  (text up to `output.inline_limit` chars, or a file path), the model id, offloaded token counts and notes.
- **B, subagent.** The main session hands a whole job to `local-worker` (`~/.claude/agents/local-worker.md`,
  runs on Haiku, has only the `lm_*` tools and `Glob`). The subagent chains several tool calls, typically
  `lm_search` to locate files and then `lm_delegate` on them; all intermediate results stay in the subagent's
  context and only its final answer returns to the main session. Use it when the job needs more than one
  tool call or when the main context must stay as small as possible.
- **C, CLI.** Same core, for scripts and manual use. `--json` prints the full result object.

### One delegation call, step by step

1. **Config.** Layers merged in this order: package default, `~/.lmagent/config.yaml`, `./lmagent.yaml`,
   `LMAGENT_CONFIG`/`--config`. The project directory (`cwd` argument or the working directory) is fixed
   here; every relative path below resolves against it.
2. **Inputs.** Each entry in `files` may be a file, a directory (walked recursively) or a glob. Dropped:
   binaries (NUL byte in the first 8 KB), files over `chunking.max_file_bytes`, and anything under `.git`,
   `node_modules`, `__pycache__`, `.venv`, `.lmagent`, `dist`, `build`, `.next`. Inline `text` is appended as
   one more item. Skips are reported in `notes`.
3. **Task template.** Picks the role (`code`, `text`, `bulk`), the processing mode, the system and user
   prompts, the reduce prompt, an optional JSON schema, and `output_ratio` (expected output size relative to
   the input, e.g. 1.2 for translate/rewrite, 0.15 for summarize).
4. **Model.** An explicit `model` wins. Otherwise, with `load.policy: smart`: if the role model is loaded or
   nothing is loaded, use the role model; if another LLM is loaded, switch only when the input is at least
   `load.switch_min_tokens`, otherwise reuse the loaded one (noted in `notes`). `strict` always uses the role
   model, `prefer_loaded` never switches. A switch unloads other LLMs and calls `POST /api/v1/models/load`
   with `context_length`, `ttl_seconds` and `parallel`; the load time is recorded.
5. **Plan.** Reads the loaded context length and slot count from `GET /api/v1/models` and derives the number
   of workers and the chunk budget so that, per worker,
   `chunk x token_safety + output + overhead <= ctx / workers`, where output is the larger of
   `generation.reserve_tokens` and `chunk x token_safety x output_ratio`, and output must also fit
   `generation.max_tokens`. Workers are reduced only when that actually buys a bigger chunk.
6. **Chunking.** Line-based split under the budget with `chunking.overlap_tokens` of overlap. For
   non-per-file modes all inputs are first joined with `### FILE: path` headers (a single input has no header).
7. **Mode.** See the branches below.
8. **Call.** `POST /v1/chat/completions` with system + user prompt, `temperature`, `max_tokens`,
   `reasoning_effort: none` (unless `generation.thinking: true`) and, for schema tasks, a strict JSON
   `response_format`. Retries on timeouts, connection errors, 5xx and "model not loaded" (the model is
   reloaded first). If the server answers "context size exceeded", the chunk is split in half and both halves
   are retried. `<think>` blocks are stripped from the answer.
9. **Output.** `rewrite` writes each file to `.lmagent/out/<path>` or, with `in_place`, over the source.
   An explicit `output_file` is written as given. A result longer than `output.inline_limit` is also written
   to `.lmagent/out/<timestamp>_<task>.<ext>`.
10. **Log.** One JSON line per run in `~/.lmagent/log.jsonl`: task, model, calls, tokens in/out, wall time,
    load time, output path. `lmagent stats` aggregates it.
11. **Return.** CLI prints the text; the MCP tool returns `{result, truncated, output_path, files_written,
    model, offloaded_tokens, calls, elapsed_s, notes}` or `{error}`.

### Mode branches

- **`auto`** (`ask`, `explain_diff`): if the joined input fits one chunk, a single call. Otherwise the
  `map_reduce` branch.
- **`single`** (`generate`): always one call. If the input does not fit, only the first chunk is sent and a
  note says so. Output code fences around whole-file results are stripped.
- **`per_chunk`** (`translate`): every chunk is processed independently in parallel; outputs are joined in
  the original order. Chunks are sized so that the output fits `max_tokens` (ratio 1.2).
- **`per_file`** (`rewrite`, `classify`): each file is processed on its own, files in parallel.
  - `rewrite` (sub-mode `per_chunk`): a file larger than the budget is processed chunk by chunk and
    reassembled; the result is the complete new file content, written to disk, and the returned text is the
    list `path -> written path`.
  - `classify` (sub-mode `single`, JSON schema): one call per file, truncated to the first chunk if huge;
    results are merged into one JSON array `[{file, label, confidence, reason}]`.
- **`map_reduce`** (`summarize`, `extract`, and `auto` overflow): chunks are processed in parallel into
  partial answers. Partials are then grouped into batches that fit the budget and reduced with the task's
  reduce prompt (`MERGE_SYSTEM`); if more than one batch remains, the reduction repeats on the batch results
  until one answer is left. `extract` reduces JSON arrays into one deduplicated array.
- **`lm_batch`**: independent jobs run concurrently, each through the full pipeline above, up to 4 at once.

### Search branch (`lm_search`, `lmagent search`)

1. Load `.lmagent/index/` if present (`meta.json` + `vectors.npy`); an index built with a different
   embedding model or chunk size is rebuilt.
2. Walk the root (same skip rules as above, plus `index.exclude` patterns). Files whose size and mtime are
   unchanged keep their vectors; new or changed files are re-chunked (`index.chunk_tokens`, line ranges kept)
   and embedded in batches with the `search_document:` prefix; deleted files are dropped.
3. Make sure the embedding model is loaded (it JIT-loads next to the LLM and is never swapped out), save
   the index.
4. Embed the query with the `search_query:` prefix, rank all chunks by cosine similarity, return the top `k`
   as `{file, start_line, end_line, score, text}`; with `files_only`, the best chunk per file.
5. Typical chain: `lm_search` to find where something lives, then `lm_delegate` with those files.

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
