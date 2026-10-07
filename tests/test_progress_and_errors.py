"""Progress callback, the friendly server-down error, and the example configs."""
import time
from pathlib import Path

import pytest
import yaml

from lmagent.client import LMStudioClient, LMStudioDown
from lmagent.config import load_config
from lmagent.runner import Runner

from conftest import FakeClient

DEAD = "http://127.0.0.1:9"  # discard port: connection refused immediately


def test_progress_ticks_per_model_call_and_total_grows(cfg):
    cfg["chunking"]["chunk_tokens"] = 500
    text = "".join(f"line {i} " + "word " * 20 + "\n" for i in range(300))  # several chunks
    client = FakeClient(reply=lambda m: "- fact")
    ticks: list[tuple[int, int, str]] = []
    r = Runner(cfg, client=client)
    res = r.run("summarize", instruction="sum", text=text,
                on_progress=lambda d, t, m: ticks.append((d, t, m)))
    assert res.calls == len(client.calls) == len(ticks)
    assert [d for d, _, _ in ticks] == list(range(1, len(ticks) + 1))
    assert all(t >= d for d, t, _ in ticks)
    assert ticks[-1][0] == ticks[-1][1]  # the estimate is exact at the end
    assert ticks[-1][2].startswith("reduce")
    assert any(m.startswith("chunk") for _, _, m in ticks)


def test_progress_per_file_labels_files(cfg, project):
    for name in ("a.py", "b.py"):
        (project / name).write_text(f"# {name}\nx = 1\n")
    ticks = []
    Runner(cfg, client=FakeClient(reply=lambda m: "x = 2\n")).run(
        "rewrite", instruction="bump", files=["a.py", "b.py"],
        on_progress=lambda d, t, m: ticks.append((d, t, m)))
    assert sorted(m for _, _, m in ticks) == ["a.py", "b.py"]
    assert ticks[-1][:2] == (2, 2)


def test_broken_progress_sink_does_not_fail_the_job(cfg):
    def boom(*a):
        raise RuntimeError("sink down")
    res = Runner(cfg, client=FakeClient()).run("ask", instruction="q", text="hello", on_progress=boom)
    assert res.text == "ok"


def test_server_down_is_one_friendly_error_without_retries():
    c = LMStudioClient(DEAD, timeout=5, retries=3, retry_delay=5)
    t0 = time.time()
    with pytest.raises(LMStudioDown) as e:
        c.models()
    assert "Start it" in str(e.value) and DEAD in str(e.value)
    with pytest.raises(LMStudioDown):
        c.chat("m", [{"role": "user", "content": "hi"}])
    with pytest.raises(LMStudioDown):
        c.embed("m", ["x"])
    # Windows needs ~2 s to report a refused localhost connection; a single retry would add 5 s per call
    assert time.time() - t0 < 3 * 4


def test_runner_surfaces_server_down(cfg):
    cfg["server"]["base_url"] = DEAD
    with pytest.raises(LMStudioDown):
        Runner(cfg).run("ask", instruction="q", text="hello")


@pytest.mark.parametrize("name", ["log-heavy.yaml", "monorepo.yaml", "translation.yaml"])
def test_example_configs_load(name, project, monkeypatch):
    from lmagent import config as config_mod
    monkeypatch.setattr(config_mod, "USER_PATH", project / "none.yaml")
    path = Path(__file__).resolve().parents[1] / "examples" / name
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg = load_config(explicit=str(path), cwd=project)
    for section, values in raw.items():
        for key, value in values.items():
            assert cfg[section][key] == value
    assert cfg["models"]["embed"]  # defaults still present
