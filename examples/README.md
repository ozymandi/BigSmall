# Per-project `lmagent.yaml` examples

Copy one of these into the project root as `lmagent.yaml` (or merge the keys you need). It is layered
on top of the package defaults and `~/.lmagent/config.yaml`; lists such as `index.exclude` are replaced,
not merged, so repeat the default entries you want to keep.

| File | When |
|---|---|
| `log-heavy.yaml` | repos with big logs or data dumps: smaller chunks, more parallel workers, logs out of the search index |
| `monorepo.yaml` | vendored forks and generated code: keep them out of the index, fewer hits per answer |
| `translation.yaml` | documentation repos where translation quality matters more than speed: Gemma for the `text` role |
