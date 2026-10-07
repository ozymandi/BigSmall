from lmagent.chunker import count_tokens, split_text
from lmagent.index import split_lines


def lines(n: int, width: int = 8) -> str:
    return "".join(f"line {i:04d} " + "x " * width + "\n" for i in range(n))


def test_split_text_small_input_is_one_chunk():
    text = lines(5)
    assert split_text(text, 1000) == [text]


def test_split_text_respects_budget_and_keeps_every_line():
    text = lines(200)
    budget = 120
    chunks = split_text(text, budget, overlap_tokens=0)
    assert len(chunks) > 1
    assert all(count_tokens(c) <= budget for c in chunks)
    assert "".join(chunks) == text  # no overlap: concatenation restores the input


def test_split_text_overlap_repeats_tail_lines():
    text = lines(200)
    chunks = split_text(text, 120, overlap_tokens=30)
    assert len(chunks) > 1
    first_tail = chunks[0].splitlines()[-1]
    assert first_tail in chunks[1].splitlines()[:3]
    # overlap never exceeds the budget either
    assert all(count_tokens(c) <= 120 for c in chunks)


def test_split_text_giant_line_is_hard_split():
    giant = "word " * 5000  # one line, far over any budget
    chunks = split_text("a\n" + giant + "\nb\n", 300)
    assert all(count_tokens(c) <= 300 for c in chunks)
    assert "".join(chunks).replace("\n", "") == ("a" + giant + "b").replace("\n", "")


def test_split_lines_line_ranges_are_contiguous_and_one_based():
    text = lines(60)
    chunks = split_lines(text, chunk_tokens=100, overlap_tokens=0)
    assert chunks[0][0] == 1
    assert chunks[-1][1] == 60
    for (s1, e1, _), (s2, _, _) in zip(chunks, chunks[1:]):
        assert s2 == e1 + 1
    for s, e, body in chunks:
        assert body.count("\n") == e - s + 1
        assert body.startswith(f"line {s - 1:04d}")


def test_split_lines_overlap_rewinds_start():
    text = lines(60)
    chunks = split_lines(text, chunk_tokens=100, overlap_tokens=25)
    assert len(chunks) > 1
    assert chunks[1][0] <= chunks[0][1]  # second chunk starts inside the first one
