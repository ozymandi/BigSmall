"""Intake: link parsing, task and project flows with a scripted agent client, files written."""
import json
from pathlib import Path

import pytest

from lmagent.client import AgentResult, ChatResult
from lmagent.intake import Intake, final_digest, parse_link

from conftest import FakeClient, QWEN


def test_parse_task_link_with_comment_anchor():
    l = parse_link("https://ga6814.worksection.com/project/348940/22729222/#com27683153")
    assert (l.project_id, l.task_id, l.comment_id) == ("348940", "22729222", "27683153")


def test_parse_project_and_subtask_links():
    assert parse_link("https://x.worksection.com/project/348940/").task_id is None
    assert parse_link("https://x.worksection.com/project/348940").project_id == "348940"
    assert parse_link("https://x.worksection.com/project/1/2/3/?x=1").task_id == "3"
    with pytest.raises(ValueError):
        parse_link("https://x.worksection.com/tasks/")


class AgentFake(FakeClient):
    """FakeClient plus chat_agent: scripted tool calls and digest per call."""

    def __init__(self, script):
        super().__init__(reply=lambda m: "## merged\n- all facts once")
        self.script = list(script)
        self.agent_calls: list[dict] = []

    def chat_agent(self, model, input, integrations, system_prompt="", reasoning="off", **kw):
        self.agent_calls.append({"input": input, "integrations": integrations, "system": system_prompt})
        calls, text = self.script.pop(0)
        return AgentResult(text=text, model=model, tool_calls=calls, prompt_tokens=100, completion_tokens=20)


def test_task_flow_writes_digest_raw_and_run(cfg):
    calls = [{"tool": "get_task", "arguments": {"task_id": "22729222"}, "output": '{"id": 22729222}'},
             {"tool": "get_file_content", "arguments": {"file_id": "9"}, "output": "brief text"}]
    client = AgentFake([(calls, "## Клієнт і контекст\n- ACME (task)")])
    res = Intake(cfg, client=client).run("https://a.worksection.com/project/348940/22729222/#com1")

    assert res.digest.startswith("## Клієнт")
    assert Path(res.digest_path) == Path(cfg["_cwd"]) / "intake" / "digest.md"
    body = Path(res.digest_path).read_text(encoding="utf-8")
    assert body.startswith("<!-- lmagent intake") and body.rstrip().endswith("(task)")
    raw = sorted(p.name for p in Path(res.raw_dir).iterdir())
    assert raw == ["01_get_task.json", "02_get_file_content.json"]
    assert json.loads((Path(res.raw_dir) / "02_get_file_content.json").read_text(encoding="utf-8"))["output"] == "brief text"
    run = json.loads((Path(res.digest_path).parent / "run.json").read_text(encoding="utf-8"))
    assert run["tool_calls"][1]["output_chars"] == len("brief text") and "output" not in run["tool_calls"][1]
    # the prompt names the task and the tools, the integration is the configured MCP with a whitelist
    call = client.agent_calls[0]
    assert "Task id: 22729222" in call["input"] and "get_task_discussion" in call["input"]
    assert call["integrations"][0]["id"] == "mcp/worksection"
    assert "get_file_content" in call["integrations"][0]["allowed_tools"]
    assert res.prompt_tokens == 100 and res.calls == 1


def test_project_flow_digests_each_task_then_merges(cfg):
    cfg["intake"]["max_tasks"] = 2
    overview_calls = [{"tool": "get_project", "arguments": {}, "output": '{"id": 348940, "name": "Site"}'},
                      {"tool": "get_tasks", "arguments": {}, "output":
                       '[{"id": 1, "title": "Logo"}, {"id": 2, "title": "Landing"}, {"id": 3, "title": "Extra"}]'}]
    client = AgentFake([(overview_calls, "overview"),
                        ([{"tool": "get_task", "arguments": {}, "output": "t1"}], "digest 1"),
                        ([{"tool": "get_task", "arguments": {}, "output": "t2"}], "digest 2")])
    ticks = []
    res = Intake(cfg, client=client).run("https://a.worksection.com/project/348940/",
                                         on_progress=lambda d, t, m: ticks.append((d, t, m)))
    assert len(client.agent_calls) == 3                       # overview + 2 tasks (max_tasks cap)
    assert "Task id: 1." in client.agent_calls[1]["input"] and "Task id: 2." in client.agent_calls[2]["input"]
    assert res.digest.startswith("## merged")                  # the merge went through chat()
    assert len(client.calls) == 1 and "digest 1" in client.calls[0][-1]["content"]
    assert ticks[-1][0] == ticks[-1][1] and any("merge" in m for _, _, m in ticks)
    assert res.calls == 3 and len(res.tool_calls) == 4


def test_empty_digest_is_noted(cfg):
    client = AgentFake([([], "")])
    res = Intake(cfg, client=client).run("https://a.worksection.com/project/1/2/")
    assert any("no digest" in n for n in res.notes)


def test_final_digest_drops_narration_and_uses_last_message():
    last = "Let me read the discussion.\n\n## Клієнт і контекст\n- x"
    r = AgentResult(text=last, model=QWEN, messages=["I'll start with get_task.", last])
    assert final_digest(r) == "## Клієнт і контекст\n- x"
    assert final_digest(AgentResult(text="no headings here", model=QWEN)) == "no headings here"


def test_chat_agent_parses_blocks(monkeypatch):
    from lmagent.client import LMStudioClient
    import httpx
    c = LMStudioClient("http://127.0.0.1:9", timeout=1, api_key="k")
    payload = {"model_instance_id": QWEN, "output": [
        {"type": "message", "content": "Working..."},
        {"type": "tool_call", "tool": "get_task", "arguments": {"task_id": "1"}, "output": "{}"},
        {"type": "reasoning", "content": "hmm"},
        {"type": "invalid_tool_call", "reason": "bad json", "metadata": {}},
        {"type": "message", "content": "## Digest\n- done"}],
        "stats": {"input_tokens": 10, "total_output_tokens": 5}}
    monkeypatch.setattr(c, "_post_with_retry", lambda body, model, path="": httpx.Response(200, json=payload))
    r = c.chat_agent(QWEN, "go", [{"type": "plugin", "id": "mcp/x"}])
    assert r.text == "## Digest\n- done" and r.messages == ["Working...", "## Digest\n- done"]
    assert [t["tool"] for t in r.tool_calls] == ["get_task"] and r.invalid[0]["reason"] == "bad json"
    assert (r.prompt_tokens, r.completion_tokens) == (10, 5)
