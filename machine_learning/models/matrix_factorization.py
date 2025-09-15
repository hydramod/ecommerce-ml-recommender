from __future__ import annotations
import numpy as np
from scipy import sparse

class ImplicitALS:
    """
    Thin wrapper around implicit.als.AlternatingLeastSquares with BM25 weighting.
    """
    def __init__(self, factors=128, iterations=20, regularization=0.01, alpha=40.0):
        from implicit.als import AlternatingLeastSquares
        self._ALS = AlternatingLeastSquares(
            factors=factors, iterations=iterations, regularization=regularization
        )
        self.alpha = alpha
        self.model = None

    def fit(self, user_item_csr: sparse.csr_matrix):
        # BM25 or TF-IDF weighting improves ALS on implicit data
        from implicit.nearest_neighbours import bm25_weight
        Cui = bm25_weight(user_item_csr, K1=100, B=0.8).tocsr() * self.alpha
        self.model = self._ALS
        self.model.fit(Cui)

    def recommend(self, user_index: int, k: int = 10, seen=None) -> np.ndarray:
        if seen is None:
            seen = []
        ids, _ = self.model.recommend(userid=user_index, user_items=None, N=k+len(seen), filter_items=seen)
        # implicit already filters via filter_items; still trim to k
        return np.array(ids[:k], dtype=int)

    def save(self, path_npz):
        if self.model is None:
            return
        np.savez_compressed(
            path_npz,
            user_factors=self.model.user_factors,
            item_factors=self.model.item_factors,
        )

    @classmethod
    def load(cls, path_npz) -> "ImplicitALS":
        data = np.load(path_npz)
        obj = cls()
        # Rebuild a minimal ALS-like stub so .recommend works via dot products
        class _Stub:
            def __init__(self, uf, vf):
                self.user_factors = uf
                self.item_factors = vf
            def recommend(self, userid, user_items, N, filter_items=None):
                import numpy as np
                u = self.user_factors[userid]
                scores = self.item_factors @ u
                if filter_items:
                    scores[np.array(list(filter_items), dtype=int)] = -1e9
                topk = np.argpartition(-scores, kth=min(N, len(scores)-1))[:N]
                topk = topk[np.argsort(-scores[topk])]
                return topk, scores[topk]
        stub = _Stub(data["user_factors"], data["item_factors"])
        obj.model = stub
        return obj
