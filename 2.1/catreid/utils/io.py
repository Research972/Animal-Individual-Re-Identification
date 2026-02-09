from pathlib import Path
import json
import torch

def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)

def save_checkpoint(path: str, model_state: dict, meta: dict):
    obj = {"model": model_state, "meta": meta}
    torch.save(obj, path)

def save_json(path: str, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
