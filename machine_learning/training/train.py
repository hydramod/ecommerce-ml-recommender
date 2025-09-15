import argparse, json, os, time
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from scipy import sparse

# Optional: uses your dual loader if present
try:
    from ml.data.loaders import load_data
except Exception:
    load_data = None

from ml.models.baselines import PopularityRecommender
from ml.models.matrix_factorization import ImplicitALS
from ml.training.eval import recall_at_k, ndcg_at_k


def _load_cfg(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _load_interactions(cfg) -> pd.DataFrame:
    # Prefer your loaders.py (hybrid). Otherwise read parquet directly for files backend.
    if load_data is not None:
        inter, _ = load_data(cfg)
        return inter

    backend = cfg["data"].get("backend", "files")
    if backend == "files":
        p = cfg["data"]["interactions_path"]
        df = pd.read_parquet(p)
        return df[["user_id", "item_id", "event_time", "event_type"]]
    else:
        raise RuntimeError("Trino backend requires ml/data/loaders.py")

def _prep(df: pd.DataFrame) -> pd.DataFrame:
    # Normalize types
    df = df.dropna(subset=["user_id","item_id","event_time"])
    df["user_id"] = df["user_id"].astype(str)
    df["item_id"] = df["item_id"].astype(str)
    df["event_time"] = pd.to_datetime(df["event_time"], utc=True)
    if "event_type" not in df.columns:
        df["event_type"] = "purchase"
    return df

def _temporal_split(df: pd.DataFrame, val_days: int, test_days: int):
    # Split by time windows relative to max timestamp
    max_ts = df["event_time"].max()
    test_start = max_ts - pd.Timedelta(days=test_days)
    val_start  = test_start - pd.Timedelta(days=val_days)

    train = df[df["event_time"] < val_start]
    val   = df[(df["event_time"] >= val_start) & (df["event_time"] < test_start)]
    test  = df[df["event_time"] >= test_start]
    return train, val, test

def _make_mappings(df: pd.DataFrame):
    users = df["user_id"].unique()
    items = df["item_id"].unique()
    user_to_idx = {u:i for i,u in enumerate(users)}
    item_to_idx = {i:j for j,i in enumerate(items)}
    idx_to_user = {i:u for u,i in user_to_idx.items()}
    idx_to_item = {j:i for i,j in item_to_idx.items()}
    return user_to_idx, item_to_idx, idx_to_user, idx_to_item

def _to_csr(df: pd.DataFrame, user_to_idx, item_to_idx):
    # simple implicit feedback weight = 1 per event (extend later if you add views/cart)
    rows = df["user_id"].map(user_to_idx).values
    cols = df["item_id"].map(item_to_idx).values
    data = np.ones(len(df), dtype=np.float32)
    shape = (len(user_to_idx), len(item_to_idx))
    return sparse.coo_matrix((data, (rows, cols)), shape=shape, dtype=np.float32).tocsr()

def _eval_model(name, model, train_csr, test_inter, user_to_idx, item_to_idx, k):
    # Build ground truth per user for test split
    gt = {}
    for r in test_inter.itertuples(index=False):
        ui = user_to_idx.get(r.user_id)
        ii = item_to_idx.get(r.item_id)
        if ui is None or ii is None:  # unseen in train universe
            continue
        gt.setdefault(ui, set()).add(ii)

    users = list(gt.keys())
    if not users:
        return {"users_evaluated": 0, "recall@{}".format(k): 0.0, "ndcg@{}".format(k): 0.0}

    recalls, ndcgs = [], []
    for u in users:
        recs = model.recommend(u, k=k, seen=train_csr.indices[train_csr.indptr[u]:train_csr.indptr[u+1]])
        recalls.append(recall_at_k(recs, gt[u], k))
        ndcgs.append(ndcg_at_k(recs, gt[u], k))
    return {
        "model": name,
        "users_evaluated": len(users),
        f"recall@{k}": float(np.mean(recalls)),
        f"ndcg@{k}": float(np.mean(ndcgs)),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", "-c", required=True)
    args = ap.parse_args()

    cfg = _load_cfg(args.config)
    k = cfg.get("eval", {}).get("k", 10)
    val_days = cfg.get("eval", {}).get("split", {}).get("val_days", 23)
    test_days = cfg.get("eval", {}).get("split", {}).get("test_days", 7)

    out_dir = Path(cfg.get("registry", {}).get("artifact_dir", "ml/artifacts/latest"))
    out_dir.mkdir(parents=True, exist_ok=True)

    print(">> Loading interactions...")
    inter = _load_interactions(cfg)
    inter = _prep(inter)

    print(">> Temporal split...")
    train_df, val_df, test_df = _temporal_split(inter, val_days, test_days)

    print(">> Building ID mappings and CSR...")
    user_to_idx, item_to_idx, idx_to_user, idx_to_item = _make_mappings(train_df)
    train_csr = _to_csr(train_df, user_to_idx, item_to_idx)

    # ----- Popularity baseline -----
    print(">> Training Popularity baseline...")
    pop = PopularityRecommender.from_events(train_df, user_to_idx, item_to_idx)
    pop_metrics = _eval_model("popularity", pop, train_csr, test_df, user_to_idx, item_to_idx, k)
    print(pop_metrics)

    # ----- Implicit ALS -----
    print(">> Training Implicit ALS...")
    als_params = cfg.get("model", {})
    als = ImplicitALS(
        factors=int(als_params.get("factors", 128)),
        iterations=int(als_params.get("iters", 20)),
        regularization=float(als_params.get("reg", 0.01)),
        alpha=float(als_params.get("alpha", 40.0)),
    )
    als.fit(train_csr)
    als_metrics = _eval_model("implicit_als", als, train_csr, test_df, user_to_idx, item_to_idx, k)
    print(als_metrics)

    # ----- choose best (by ndcg@k) -----
    best = als if als_metrics[f"ndcg@{k}"] >= pop_metrics[f"ndcg@{k}"] else pop
    best_name = "implicit_als" if best is als else "popularity"

    # ----- persist artifacts -----
    print(f">> Saving artifacts to {out_dir} (best={best_name})")
    (out_dir / "mappings").mkdir(exist_ok=True)
    with open(out_dir / "mappings" / "user_to_idx.json", "w") as f:
        json.dump(user_to_idx, f)
    with open(out_dir / "mappings" / "item_to_idx.json", "w") as f:
        json.dump(item_to_idx, f)

    # Save models
    pop.save(out_dir / "popularity.json")
    als.save(out_dir / "als.npz")

    metrics = {
        "timestamp": int(time.time()),
        "k": k,
        "popularity": pop_metrics,
        "implicit_als": als_metrics,
        "best_model": best_name,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Minimal service config the API can read
    with open(out_dir / "model_meta.json", "w") as f:
        json.dump({"production": best_name}, f)

    print(">> Done.")

if __name__ == "__main__":
    main()
