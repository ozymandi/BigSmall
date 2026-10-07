"""Attachment download: parsing the saved tool outputs, docs/all/none filter, size cap, names, errors."""
import json
from pathlib import Path

from lmagent.wsfiles import Attachment, attachments_from_raw, bytes_from_contents, download, select


def _raw(tmp: Path, tool: str, payload: dict | list, n: int = 1) -> None:
    text = json.dumps(payload)
    blocks = json.dumps([{"type": "text", "text": text}])
    (tmp / f"{n:02d}_{tool}.json").write_text(json.dumps({"tool": tool, "arguments": {}, "output": blocks}),
                                              encoding="utf-8")


def test_attachments_from_raw_reads_task_and_comment_files_once(tmp_path):
    _raw(tmp_path, "get_all_task_attachments", {
        "task_files": [{"id": 1, "size": "500", "name": "brief.pdf", "page": "/download/1/"}],
        "comment_files": [{"id": 2, "size": "79299367", "name": "Landing page.mov", "comment_id": 9},
                          {"id": 1, "size": "500", "name": "brief.pdf"}]}, 1)
    _raw(tmp_path, "get_task", {"id": 5, "name": "not a file list"}, 2)
    _raw(tmp_path, "get_project_files", [{"id": "7", "size": 12, "name": "logo.svg"}], 3)
    atts = attachments_from_raw(tmp_path)
    assert [(a.id, a.name, a.size, a.origin) for a in atts] == [
        ("1", "brief.pdf", 500, "task"), ("2", "Landing page.mov", 79299367, "comment"), ("7", "logo.svg", 12, "project")]


def test_select_modes_and_size_cap():
    atts = [Attachment("1", "brief.pdf", 500), Attachment("2", "clip.mov", 79_000_000),
            Attachment("3", "photos.zip", 16_500_000), Attachment("4", "shot.png", 2_000_000)]
    keep, skipped = select(atts, "docs", 30)
    assert [a.name for a in keep] == ["brief.pdf", "photos.zip"]
    assert {s["name"]: s["reason"] for s in skipped} == {"clip.mov": "not a document", "shot.png": "not a document"}
    keep, skipped = select(atts, "all", 30)
    assert [a.name for a in keep] == ["brief.pdf", "photos.zip", "shot.png"] and skipped[0]["reason"] == "over 30 MB"
    keep, skipped = select(atts, "none", 30)
    assert keep == [] and len(skipped) == 4


def test_download_writes_files_handles_duplicates_and_errors(tmp_path):
    atts = [Attachment("1", "brief.pdf", 3), Attachment("2", "brief.pdf", 3),
            Attachment("3", "bad:name?.txt", 2), Attachment("4", "broken.pdf", 1)]

    def reader(uri: str) -> bytes:
        fid = uri.rsplit("/", 1)[1]
        if fid == "4":
            raise RuntimeError("boom")
        return f"f{fid}".encode()

    ticks = []
    rep = download(atts, tmp_path / "files", "docs", 30, reader, on_progress=lambda d, t, m: ticks.append((d, t)))
    names = sorted(p.name for p in (tmp_path / "files").iterdir())
    assert names == ["bad_name_.txt", "brief.pdf", "brief_2.pdf"]
    assert (tmp_path / "files" / "brief_2.pdf").read_bytes() == b"f2"
    assert [s["name"] for s in rep.saved] == ["brief.pdf", "brief_2.pdf", "bad_name_.txt"]
    assert rep.errors == [{"id": "4", "name": "broken.pdf", "error": "RuntimeError: boom"}]
    assert ticks[-1] == (4, 4)
    sec = rep.section("intake/files")
    assert sec.startswith("## Файли на диску") and "помилка: broken.pdf" in sec and "brief_2.pdf" in sec


def test_download_none_touches_nothing(tmp_path):
    rep = download([Attachment("1", "a.pdf", 1)], tmp_path / "files", "none", 30, lambda u: b"x")
    assert not (tmp_path / "files").exists() and rep.skipped[0]["reason"] == "download=none"


def test_bytes_from_contents_unwraps_fastmcp_json_text():
    import base64
    b64 = base64.b64encode(b"%PDF-1.4 x").decode()
    # FastMCP: the server's dict comes back as JSON text
    assert bytes_from_contents([(None, json.dumps({"uri": "worksection://file/1", "mimeType": "application/pdf", "blob": b64}))]) == b"%PDF-1.4 x"
    assert bytes_from_contents([(None, json.dumps({"uri": "u", "mimeType": "text/plain", "text": "hi"}))]) == b"hi"
    # native blob / plain text contents
    assert bytes_from_contents([(b64, None)]) == b"%PDF-1.4 x"
    assert bytes_from_contents([(None, "plain text")]) == b"plain text"
    assert bytes_from_contents([(None, json.dumps({"other": 1}))]) == json.dumps({"other": 1}).encode()
