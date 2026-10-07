import os
import time

from lmagent.index import Index, kind_of

from conftest import FakeClient


def write(p, text):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    # make sure a later rewrite gets a different mtime even on coarse filesystems
    os.utime(p, (time.time() - 10, time.time() - 10))


def make_index(cfg, project, client=None):
    client = client or FakeClient()
    return Index(cfg, client, project), client


def test_update_is_incremental(cfg, project):
    write(project / "a.py", "def alpha():\n    return 1\n")
    write(project / "b.py", "def beta():\n    return 2\n")
    write(project / "README.md", "# readme\nsome words\n")
    idx, client = make_index(cfg, project)
    s = idx.update()
    assert s["indexed"] == 3 and s["total_files"] == 3
    embedded_first = len(client.embedded)

    # nothing changed: no embedding calls, everything kept
    idx2, client2 = make_index(cfg, project)
    s = idx2.update()
    assert s["indexed"] == 0 and s["unchanged"] == 3 and client2.embedded == []

    # one changed, one deleted, one new
    (project / "b.py").unlink()
    write(project / "a.py", "def alpha():\n    return 42\n")
    write(project / "c.py", "def gamma():\n    return 3\n")
    idx3, client3 = make_index(cfg, project)
    s = idx3.update()
    assert s["indexed"] == 2 and s["removed"] == 1 and s["unchanged"] == 1
    assert set(idx3.files) == {"a.py", "c.py", "README.md"}
    assert idx3.vectors.shape[0] == sum(len(f["chunks"]) for f in idx3.files.values())
    # row ranges are consistent with the vector matrix
    rows = sorted(f["rows"] for f in idx3.files.values())
    assert rows[0][0] == 0 and rows[-1][1] == idx3.vectors.shape[0]
    for (_, e1), (s2, _) in zip(rows, rows[1:]):
        assert s2 == e1
    assert embedded_first == 3


def test_excluded_files_and_dirs_are_skipped(cfg, project):
    cfg["index"]["exclude"] = ["*.lock", "vendor/", "gen/*"]
    write(project / "keep.py", "x = 1\n")
    write(project / "poetry.lock", "lock\n")
    write(project / "vendor" / "lib" / "big.js", "var a = 1;\n")
    write(project / "gen" / "out.ts", "export const a = 1;\n")
    write(project / "src" / "vendor.py", "y = 2\n")  # a file named vendor is not a vendor/ dir
    write(project / "node_modules" / "m" / "i.js", "skip me\n")
    idx, _ = make_index(cfg, project)
    idx.update()
    assert set(idx.files) == {"keep.py", "src/vendor.py"}


def test_previously_indexed_file_is_dropped_once_excluded(cfg, project):
    write(project / "a.py", "x = 1\n")
    write(project / "notes.txt", "hello\n")
    idx, _ = make_index(cfg, project)
    idx.update()
    assert "notes.txt" in idx.files
    cfg["index"]["exclude"] = ["*.txt"]
    idx2, _ = make_index(cfg, project)
    s = idx2.update()
    assert s["removed"] == 1 and set(idx2.files) == {"a.py"}


def test_search_returns_line_ranges_and_kind_filter(cfg, project):
    write(project / "retry.py", "def retry_request():\n    pass\n")
    write(project / "docs.md", "retry request retry request retry request\n")
    idx, _ = make_index(cfg, project)
    idx.update()
    hits = idx.search("retry request", k=5)
    assert {h["file"] for h in hits} == {"retry.py", "docs.md"}
    assert all(h["start_line"] == 1 for h in hits)
    code = idx.search("retry request", k=5, kind="code")
    assert [h["file"] for h in code] == ["retry.py"]
    docs = idx.search("retry request", k=5, kind="docs", files_only=True)
    assert [h["file"] for h in docs] == ["docs.md"] and "text" not in docs[0]


def test_kind_of():
    assert kind_of("README.md") == "docs"
    assert kind_of("a/b.rst") == "docs"
    assert kind_of("x.py") == "code"
    assert kind_of("Makefile") == "code"
