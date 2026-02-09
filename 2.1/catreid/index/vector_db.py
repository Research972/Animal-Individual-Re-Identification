# catreid/index/vector_db.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import torch


@dataclass
class VectorDB:
    """
    Very simple vector DB stored as a single .pt:
      {
        "embs": [N,D] float32 CPU tensor,
        "ids":  list[str] length N,
        "paths":list[str] length N,
        "next_id": int
      }
    """
    path: str
    emb_dim: int = 256

    def __post_init__(self):
        self.path = str(self.path)
        self._embs = torch.empty((0, self.emb_dim), dtype=torch.float32)
        self._ids: List[str] = []
        self._paths: List[str] = []
        self._next_id = 0

    @property
    def size(self) -> int:
        return int(self._embs.size(0))

    def load_or_init(self):
        p = Path(self.path)
        if not p.exists():
            # init empty
            self.save()
            return
        obj = torch.load(self.path, map_location="cpu")
        self._embs = obj.get("embs", self._embs).float().cpu()
        self._ids = list(obj.get("ids", []))
        self._paths = list(obj.get("paths", []))
        self._next_id = int(obj.get("next_id", 0))

        # basic sanity
        if self._embs.dim() != 2 or self._embs.size(1) != self.emb_dim:
            raise ValueError(f"VectorDB embs shape mismatch: {tuple(self._embs.shape)} vs emb_dim={self.emb_dim}")
        if len(self._ids) != self.size or len(self._paths) != self.size:
            raise ValueError("VectorDB ids/paths length mismatch with embs")

    def save(self):
        obj = {
            "embs": self._embs.contiguous(),
            "ids": self._ids,
            "paths": self._paths,
            "next_id": self._next_id,
        }
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(obj, self.path)

    def as_tensors(self) -> Tuple[torch.Tensor, List[str], List[str]]:
        return self._embs, self._ids, self._paths

    def create_new_identity(self) -> str:
        new_id = f"id_{self._next_id:03d}"
        self._next_id += 1
        return new_id

    def add(self, emb: torch.Tensor, identity: str, img_path: str):
        """
        emb: [D] CPU tensor float32
        """
        emb = emb.detach().float().cpu().view(1, -1)
        if emb.size(1) != self.emb_dim:
            raise ValueError(f"emb dim mismatch: {emb.size(1)} vs {self.emb_dim}")
        self._embs = torch.cat([self._embs, emb], dim=0)
        self._ids.append(str(identity))
        self._paths.append(str(img_path))