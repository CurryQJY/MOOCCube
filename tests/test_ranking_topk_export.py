import json
from pathlib import Path

import pytest
import torch


from ranking_topk_export import TopKJsonlExporter


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_exporter_writes_ranked_valid_items_with_metadata(tmp_path):
    output = tmp_path / "top20.jsonl"
    scores = torch.tensor(
        [
            [0.1, 0.9, -1e9, 0.4],
            [0.8, 0.7, 0.6, 0.5],
        ],
        dtype=torch.float32,
    )

    with TopKJsonlExporter(
        output,
        top_k=3,
        metadata={"model": "demo", "seed": 2025},
    ) as exporter:
        exporter.write_batch(
            scores,
            user_ids=[10, 11],
            target_item_ids=[3, 0],
            target_popularity=[0, 0],
        )

    rows = _read_jsonl(output)
    assert rows[0]["model"] == "demo"
    assert rows[0]["seed"] == 2025
    assert rows[0]["sample_index"] == 0
    assert rows[1]["sample_index"] == 1
    assert rows[0]["recommended_item_ids"] == [1, 3, 0]
    assert rows[0]["recommended_scores"] == pytest.approx([0.9, 0.4, 0.1])
    assert rows[0]["user_id"] == 10
    assert rows[0]["target_item_id"] == 3
    assert rows[0]["target_popularity"] == 0
    assert rows[1]["recommended_item_ids"] == [0, 1, 2]


def test_exporter_filters_masked_scores_and_allows_short_lists(tmp_path):
    output = tmp_path / "top20.jsonl"
    scores = torch.tensor([[0.3, -1e9, float("-inf")]], dtype=torch.float32)

    with TopKJsonlExporter(output, top_k=20) as exporter:
        exporter.write_batch(scores, [1], [2], [0])

    row = _read_jsonl(output)[0]
    assert row["recommended_item_ids"] == [0]
    assert row["recommended_scores"] == pytest.approx([0.3])


def test_exporter_accepts_precomputed_topk_without_reranking(tmp_path):
    output = tmp_path / "top20.jsonl"
    top_items = torch.tensor([[2, 0, 1]])
    top_scores = torch.tensor([[0.5, 0.5, 0.5]])

    with TopKJsonlExporter(output, top_k=2) as exporter:
        exporter.write_precomputed_batch(top_items, top_scores, [1], [0], [0])

    row = _read_jsonl(output)[0]
    assert row["recommended_item_ids"] == [2, 0]


def test_exporter_replaces_destination_only_after_success(tmp_path):
    output = tmp_path / "top20.jsonl"
    output.write_text("old\n", encoding="utf-8")

    with TopKJsonlExporter(output, top_k=1) as exporter:
        exporter.write_batch(torch.tensor([[0.2, 0.8]]), [1], [0], [0])
        assert output.read_text(encoding="utf-8") == "old\n"

    assert _read_jsonl(output)[0]["recommended_item_ids"] == [1]


def test_exporter_preserves_destination_when_export_fails(tmp_path):
    output = tmp_path / "top20.jsonl"
    output.write_text("old\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="stop"):
        with TopKJsonlExporter(output, top_k=1) as exporter:
            exporter.write_batch(torch.tensor([[0.2, 0.8]]), [1], [0], [0])
            raise RuntimeError("stop")

    assert output.read_text(encoding="utf-8") == "old\n"
    assert not list(tmp_path.glob("*.tmp"))
