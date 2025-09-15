from pathlib import Path
import pandas as pd

def load_interactions_files(path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    return df[["user_id","item_id","event_time","event_type"]].dropna()

def load_items_files(path: str | Path) -> pd.DataFrame:
    if not path: 
        return pd.DataFrame(columns=["item_id"])
    df = pd.read_parquet(path)
    keep = [c for c in ["item_id","category","brand","price","title","description"] if c in df.columns]
    return df[keep]

def load_interactions_trino(cfg) -> pd.DataFrame:
    import pandas as pd
    from trino.dbapi import connect
    q = f"SELECT user_id,item_id,event_time,event_type FROM {cfg['catalog']}.{cfg['schema']}.{cfg['interactions_table']}"
    with connect(host=cfg["host"], port=cfg["port"], user=cfg["user"], catalog=cfg["catalog"], schema=cfg["schema"]) as cn:
        return pd.read_sql(q, cn)

def load_items_trino(cfg) -> pd.DataFrame:
    import pandas as pd
    from trino.dbapi import connect
    if not cfg.get("items_table"): 
        return pd.DataFrame(columns=["item_id"])
    q = f"SELECT item_id,category,brand,price,title,description FROM {cfg['catalog']}.{cfg['schema']}.{cfg['items_table']}"
    with connect(host=cfg["host"], port=cfg["port"], user=cfg["user"], catalog=cfg["catalog"], schema=cfg["schema"]) as cn:
        return pd.read_sql(q, cn)

def load_data(cfg):
    if cfg["data"]["backend"] == "files":
        inter = load_interactions_files(cfg["data"]["interactions_path"])
        items = load_items_files(cfg["data"].get("items_path"))
    else:
        tr = cfg["data"]["trino"]
        inter = load_interactions_trino(tr)
        items = load_items_trino(tr)
    return inter, items
