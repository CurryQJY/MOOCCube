import json
import math
import os
from pathlib import Path
from typing import Iterable, Mapping, Optional

import torch


MASKED_SCORE_THRESHOLD = -1e8


class TopKJsonlExporter:
    def __init__(
        self,
        path,
        top_k: int = 20,
        metadata: Optional[Mapping[str, object]] = None,
    ) -> None:
        if int(top_k) < 1:
            raise ValueError("top_k must be positive")
        self.path = Path(path)
        self.top_k = int(top_k)
        self.metadata = dict(metadata or {})
        self.tmp_path = Path(str(self.path) + ".tmp")
        self._handle = None
        self._sample_index = 0

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.tmp_path.unlink(missing_ok=True)
        self._handle = self.tmp_path.open("w", encoding="utf-8", newline="\n")
        return self

    def __exit__(self, exc_type, exc, traceback):
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        if exc_type is None:
            os.replace(self.tmp_path, self.path)
        else:
            self.tmp_path.unlink(missing_ok=True)
        return False

    def write_batch(
        self,
        scores: torch.Tensor,
        user_ids: Iterable[int],
        target_item_ids: Iterable[int],
        target_popularity: Iterable[int],
    ) -> None:
        if self._handle is None:
            raise RuntimeError("exporter must be used as a context manager")
        if scores.ndim != 2:
            raise ValueError("scores must have shape [batch, items]")

        k = min(self.top_k, int(scores.size(1)))
        top_scores, top_items = torch.topk(scores, k=k, dim=1)
        self.write_precomputed_batch(
            top_items,
            top_scores,
            user_ids,
            target_item_ids,
            target_popularity,
        )

    def write_precomputed_batch(
        self,
        top_items: torch.Tensor,
        top_scores: torch.Tensor,
        user_ids: Iterable[int],
        target_item_ids: Iterable[int],
        target_popularity: Iterable[int],
    ) -> None:
        if self._handle is None:
            raise RuntimeError("exporter must be used as a context manager")
        if top_items.ndim != 2 or top_scores.ndim != 2 or top_items.shape != top_scores.shape:
            raise ValueError("precomputed Top-K tensors must have matching [batch, k] shapes")

        users = [int(value) for value in user_ids]
        targets = [int(value) for value in target_item_ids]
        popularity = [int(value) for value in target_popularity]
        batch_size = int(top_items.size(0))
        if not (len(users) == len(targets) == len(popularity) == batch_size):
            raise ValueError("batch metadata lengths must match scores")

        keep_k = min(self.top_k, int(top_items.size(1)))
        top_items = top_items[:, :keep_k]
        top_scores = top_scores[:, :keep_k]
        top_scores = top_scores.detach().cpu()
        top_items = top_items.detach().cpu()

        for row in range(batch_size):
            item_ids = []
            item_scores = []
            for item_id, score in zip(top_items[row].tolist(), top_scores[row].tolist()):
                if not math.isfinite(float(score)):
                    continue
                if float(score) <= MASKED_SCORE_THRESHOLD:
                    continue
                item_ids.append(int(item_id))
                item_scores.append(float(score))

            record = dict(self.metadata)
            record.update(
                {
                    "sample_index": self._sample_index,
                    "user_id": users[row],
                    "target_item_id": targets[row],
                    "target_popularity": popularity[row],
                    "recommended_item_ids": item_ids,
                    "recommended_scores": item_scores,
                }
            )
            self._handle.write(json.dumps(record, ensure_ascii=True, allow_nan=False) + "\n")
            self._sample_index += 1
        self._handle.flush()
