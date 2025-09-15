import math
import numpy as np
from typing import Iterable, Set

def recall_at_k(recs: Iterable[int], truth: Set[int], k: int) -> float:
    if not truth:
        return 0.0
    recs_k = list(recs)[:k]
    hits = sum(1 for r in recs_k if r in truth)
    return hits / float(len(truth))

def ndcg_at_k(recs: Iterable[int], truth: Set[int], k: int) -> float:
    recs_k = list(recs)[:k]
    dcg = 0.0
    for idx, r in enumerate(recs_k, start=1):
        if r in truth:
            dcg += 1.0 / math.log2(idx + 1)
    # Ideal DCG
    ideal_hits = min(len(truth), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return 0.0 if idcg == 0 else dcg / idcg
