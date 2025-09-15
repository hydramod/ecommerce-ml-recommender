from __future__ import annotations
from dataclasses import dataclass
import json
import numpy as np
import pandas as pd
from typing import Iterable, Optional, Sequence, Set


@dataclass
class PopularityRecommender:
    # top_items: np.ndarray of item indices sorted by freq desc
    top_items: np.ndarray

    @classmethod
    def from_events(cls, events: pd.DataFrame, user_to_idx: dict, item_to_idx: dict) -> "PopularityRecommender":
        # Count item frequency in training split
        counts = events["item_id"].map(item_to_idx).value_counts().sort_values(ascending=False)
        return cls(top_items=counts.index.to_numpy())

    def recommend(self, user_index: int, k: int = 10, seen: Optional[Sequence[int]] = None) -> np.ndarray:
        if seen is None:
            seen = []
        seen_set: Set[int] = set(seen)
        recs = [i for i in self.top_items if i not in seen_set]
        return np.array(recs[:k], dtype=int)

    def save(self, path):
        data = {"top_items": self.top_items.tolist()}
        with open(path, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path) -> "PopularityRecommender":
        with open(path, "r") as f:
            data = json.load(f)
        return cls(top_items=np.array(data["top_items"], dtype=int))
