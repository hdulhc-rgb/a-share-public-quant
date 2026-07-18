from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np


@dataclass(frozen=True)
class Split:
    split_id: str
    train: np.ndarray
    test: np.ndarray


def anchored_walk_forward(n_obs: int, min_train: int, test_size: int, step: int) -> list[Split]:
    splits: list[Split] = []
    start = min_train
    split_number = 0
    while start + test_size <= n_obs:
        splits.append(Split(f"WF{split_number:03d}", np.arange(0, start), np.arange(start, start + test_size)))
        split_number += 1
        start += step
    return splits


def combinatorial_purged_cv(n_obs: int, folds: int, test_folds: int, embargo: int, max_paths: int = 15) -> list[Split]:
    indices = np.arange(n_obs)
    fold_blocks = [block for block in np.array_split(indices, folds) if len(block)]
    splits: list[Split] = []
    for path_number, selected in enumerate(combinations(range(len(fold_blocks)), test_folds)):
        if path_number >= max_paths:
            break
        test = np.sort(np.concatenate([fold_blocks[i] for i in selected]))
        excluded = set(test.tolist())
        for i in test:
            excluded.update(range(max(0, i - embargo), min(n_obs, i + embargo + 1)))
        train = np.array([i for i in indices if i not in excluded], dtype=int)
        splits.append(Split(f"CPCV{path_number:03d}", train, test))
    return splits

