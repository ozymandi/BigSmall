"""Runner._plan: chunk budget and worker count from the loaded context (one shared budget for all
concurrent sequences), the task's output_ratio and the config. Numbers below follow from the default
config: token_safety 1.5, reserve 2048, max_tokens 8192, min_chunk 6000, chunk_tokens 10000, max_parallel 4."""
import pytest

from lmagent.runner import Runner
from lmagent.tasks import get_task

from conftest import FakeClient


def plan(cfg, task: str, context: int, parallel: int = 4):
    client = FakeClient(context=context, parallel=parallel)
    r = Runner(cfg, client=client)
    notes: list[str] = []
    budget = r._plan(client.loaded_model, get_task(task), notes)
    return budget, r._workers, notes


def test_qwen_89k_summarize_keeps_full_chunks_and_all_workers(cfg):
    budget, workers, _ = plan(cfg, "summarize", 89344)
    assert workers == 4
    assert budget == 10000  # configured chunk_tokens fits


def test_qwen_89k_rewrite_is_capped_by_max_tokens(cfg):
    # output ratio 1.2: the output must fit max_tokens -> 8192 / (1.5 * 1.2) = 4551
    budget, workers, notes = plan(cfg, "rewrite", 89344)
    assert workers == 4
    assert budget == 4551
    assert any("plan:" in n for n in notes)


def test_32k_context_drops_workers_until_chunks_are_big_enough(cfg):
    # 4 workers -> 3696 tok chunks, 3 -> 5516, 2 -> 9150 >= min_chunk: stop at 2
    budget, workers, _ = plan(cfg, "summarize", 32768)
    assert workers == 2
    assert budget == 9150


def test_workers_are_not_reduced_when_it_does_not_buy_a_bigger_chunk(cfg):
    # rewrite on 32k: 4 workers -> 2300 tok, 3 -> 3128, 2 -> 4551 (max_tokens cap), 1 -> still 4551:
    # going from 2 to 1 buys nothing, so the planner stops at 2 even though 4551 < min_chunk
    budget, workers, _ = plan(cfg, "rewrite", 32768)
    assert workers == 2
    assert budget == 4551


def test_parallel_slots_cap_workers(cfg):
    _, workers, _ = plan(cfg, "summarize", 89344, parallel=2)
    assert workers == 2


@pytest.mark.parametrize("task,ratio", [("summarize", 0.15), ("classify", 0.05), ("translate", 1.2),
                                        ("rewrite", 1.2), ("ask", 0.3)])
def test_output_ratio_per_task(task, ratio):
    assert get_task(task).output_ratio == ratio


def test_budget_never_below_floor(cfg):
    budget, workers, _ = plan(cfg, "translate", 4096)
    assert workers == 1
    assert budget >= 500
