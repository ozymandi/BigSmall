"""Output cleaning, JSON parsing, config layering and the rewrite guardrails end to end with a fake model."""
import json

import yaml

from lmagent.cli import _parse_params
from lmagent.config import load_config
from lmagent.runner import Runner, _parse_json, check_syntax, clean_output, diff_stat, strip_fences, write_like

from conftest import FakeClient


# --- small helpers -----------------------------------------------------------

def test_strip_fences_and_parse_json():
    assert strip_fences("```json\n[1, 2]\n```") == "[1, 2]"
    assert strip_fences("```\nplain\n```") == "plain"
    assert strip_fences("no fences") == "no fences"
    assert _parse_json("```json\n{\"a\": 1}\n```") == {"a": 1}
    assert _parse_json("not json") is None


def test_clean_output_drops_echoed_content_wrapper():
    assert clean_output("<content>\nabc\ndef\n</content>") == "abc\ndef"
    assert clean_output("<content>\r\nabc\r\n") == "abc"
    assert clean_output("```python\nx = 1\n```") == "x = 1"
    assert clean_output("plain <content> inside") == "plain <content> inside"


def test_check_syntax_by_extension():
    assert check_syntax("a.py", "def f(:\n")
    assert check_syntax("a.py", "def f():\n    pass\n") is None
    assert check_syntax("a.json", "{bad") and check_syntax("a.json", '{"a": 1}') is None
    assert check_syntax("a.yaml", "a: [1,\n") and check_syntax("a.yaml", "a: 1\n") is None
    assert check_syntax("a.ts", "whatever (((") is None  # no parser for ts: not checked


def test_diff_stat_counts_lines():
    assert diff_stat("a\nb\nc", "a\nx\nc\nd") == (2, 1)
    assert diff_stat("same\n", "same\n") == (0, 0)


def test_write_like_keeps_eol_and_trailing_newline(tmp_path):
    src = tmp_path / "s.py"
    src.write_bytes(b"a\r\nb\r\n")
    write_like(tmp_path / "o1", "a\nb\nc", src)
    assert (tmp_path / "o1").read_bytes() == b"a\r\nb\r\nc\r\n"
    src.write_bytes(b"a\nb")
    write_like(tmp_path / "o2", "a\nb\nc\n", src)
    assert (tmp_path / "o2").read_bytes() == b"a\nb\nc"
    write_like(tmp_path / "o3", "a\nb", None)
    assert (tmp_path / "o3").read_bytes() == b"a\nb\n"


def test_parse_params():
    assert _parse_params(["to=English", "focus=errors, warnings"]) == {"to": "English", "focus": "errors, warnings"}


# --- config layering ----------------------------------------------------------

def test_config_layers(project, monkeypatch):
    from lmagent import config as config_mod
    user = project / "user.yaml"
    user.write_text(yaml.safe_dump({"chunking": {"chunk_tokens": 7000}, "models": {"text": "user-text"}}))
    monkeypatch.setattr(config_mod, "USER_PATH", user)
    (project / "lmagent.yaml").write_text(yaml.safe_dump({"chunking": {"chunk_tokens": 3000}}))
    explicit = project / "explicit.yaml"
    explicit.write_text(yaml.safe_dump({"index": {"exclude": ["vendor/"]}}))

    c = load_config(cwd=project)
    assert c["chunking"]["chunk_tokens"] == 3000          # project beats user
    assert c["models"]["text"] == "user-text"             # user beats default
    assert c["models"]["code"] == "qwen/qwen3.8-27b"       # defaults survive a partial override
    assert c["chunking"]["overlap_tokens"] == 150
    assert c["_cwd"] == str(project)

    c2 = load_config(explicit=str(explicit), cwd=project)
    assert c2["index"]["exclude"] == ["vendor/"]          # a list is replaced, not merged
    assert c2["chunking"]["chunk_tokens"] == 3000

    monkeypatch.setenv("LMAGENT_CONFIG", str(explicit))
    assert load_config(cwd=project)["index"]["exclude"] == ["vendor/"]


# --- rewrite guardrails end to end --------------------------------------------

def rewrite_reply(messages: list[dict]) -> str:
    """Scripted model: picks the answer by what file content it was given."""
    user = messages[-1]["content"]
    if "x = 1" in user:
        return "<content>\nx = 2\n</content>"          # echoed wrapper, must be stripped
    if "y = 1" in user:
        return "```python\ndef broken(:\n    pass\n```"  # syntax error, must not be written
    if '"k": 1' in user:
        return '{"k": 1}'                               # unchanged
    return "unexpected"


def test_rewrite_in_place_guardrails(cfg, project):
    (project / "a.py").write_bytes(b"x = 1\r\n")          # CRLF source
    (project / "b.py").write_bytes(b"y = 1\n")
    (project / "c.json").write_bytes(b'{"k": 1}\n')
    r = Runner(cfg, client=FakeClient(reply=rewrite_reply))
    res = r.run("rewrite", instruction="bump", files=["a.py", "b.py", "c.json"], in_place=True)

    assert (project / "a.py").read_bytes() == b"x = 2\r\n"      # wrapper gone, CRLF kept
    assert (project / "b.py").read_bytes() == b"y = 1\n"        # original kept
    assert (project / "c.json").read_bytes() == b'{"k": 1}\n'
    assert res.files_written == [str(project / "a.py")]
    assert "a.py -> " in res.text and "(+1 -1)" in res.text
    assert "b.py: NOT written (SyntaxError" in res.text
    assert "c.json: unchanged" in res.text

    backups = list((project / ".lmagent" / "backup").rglob("*.py"))
    assert [b.name for b in backups] == ["a.py"]
    assert backups[0].read_bytes() == b"x = 1\r\n"
    assert any("backed up" in n for n in res.notes)
    assert json.loads((project / "log.jsonl").read_text().splitlines()[-1])["task"] == "rewrite"


def test_rewrite_to_out_dir_writes_even_when_check_fails(cfg, project):
    (project / "b.py").write_bytes(b"y = 1\n")
    r = Runner(cfg, client=FakeClient(reply=rewrite_reply))
    res = r.run("rewrite", instruction="bump", files=["b.py"])
    out = project / ".lmagent" / "out" / "b.py"
    assert out.read_bytes() == b"def broken(:\n    pass\n"
    assert "CHECK FAILED: SyntaxError" in res.text
    assert (project / "b.py").read_bytes() == b"y = 1\n"
    assert not (project / ".lmagent" / "backup").exists()
