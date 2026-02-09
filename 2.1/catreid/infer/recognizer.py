from pathlib import Path
import time 

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from catreid.models.resnet_embedder import ResNet50Embedder
from catreid.utils.device import get_device


class CatReIDRecognizer:
    def __init__(
        self,
        ckpt_path: str,
        vector_db_path: str,
        threshold: float = 0.25,
    ):
        self.threshold = float(threshold)

        # -------- device --------
        self.device = get_device(prefer_mps=True)

        # -------- model --------
        self.model = ResNet50Embedder(emb_dim=256, pretrained=False)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if isinstance(ckpt, dict) and "model" in ckpt:
            self.model.load_state_dict(ckpt["model"], strict=True)
        else:
            self.model.load_state_dict(ckpt, strict=True)
        self.model.to(self.device)
        self.model.eval()

        # -------- vector DB --------
        db = torch.load(vector_db_path, map_location="cpu")
        self.gallery_emb = db["embeddings"].contiguous()   # [N, D]
        if "labels" in db:
            self.gallery_ids = torch.as_tensor(db["labels"]).contiguous()
        else:
            raise KeyError("vector_db must contain 'labels'")

        self.gallery_size = int(self.gallery_emb.size(0))
        self.emb_dim = int(self.gallery_emb.size(1))

        print(f"[ReID] device={self.device}, gallery={self.gallery_size}")

        # -------- image preprocess --------
        self.tf = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    @torch.no_grad()
    def _extract_embedding(self, img_path: str):
        img = Image.open(img_path).convert("RGB")
        x = self.tf(img).unsqueeze(0).to(self.device)

        z = self.model(x)                 
        z = F.normalize(z, p=2, dim=1)     
        z_cpu = z.cpu()

        return z_cpu, float(z.norm(p=2).item())

    @torch.no_grad()
    def recognize_report(self, img_path: str, top_k: int = 5):
        t_total0 = time.perf_counter()  

        # -------- query embedding --------
        t_emb0 = time.perf_counter()    
        q, q_norm = self._extract_embedding(img_path)
        embed_time_ms = (time.perf_counter() - t_emb0) * 1000.0  

        # -------- distance computation --------
        t_search0 = time.perf_counter()  

        # cosine -> L2 等价
        sim = q @ self.gallery_emb.t()              
        dist = (2.0 - 2.0 * sim).squeeze(0)        

        k = min(top_k, self.gallery_size)
        top_dist, top_idx = torch.topk(dist, k=k, largest=False)

        search_time_ms = (time.perf_counter() - t_search0) * 1000.0          

        topk = []
        for r in range(k):
            idx = int(top_idx[r])
            topk.append({
                "rank": r + 1,
                "id": int(self.gallery_ids[idx]),
                "distance": float(top_dist[r]),
            })

        best = topk[0]
        is_match = best["distance"] < self.threshold

        # -------- decision logic --------
        if is_match:
            reason = "distance < threshold"
            action = "accept_existing_id"
            pred_id = best["id"]
            is_new = False
            new_id_suggestion = None
        else:
            reason = "distance >= threshold"
            action = "register_new_id"
            pred_id = None
            is_new = True
            new_id_suggestion = int(self.gallery_ids.max().item()) + 1

        total_time_ms = (time.perf_counter() - t_total0) * 1000.0 

        # -------- report --------
        report = {
            "img": img_path,
            "gallery_size": self.gallery_size,
            "threshold": self.threshold,
            "top_k": k,
            "embedding_dim": self.emb_dim,
            "embedding_norm": q_norm,
            "topk": topk,
            "match": bool(is_match),
            "is_new": bool(is_new),
            "reason": reason,
            "pred_id": pred_id,
            "pred_distance": float(best["distance"]),
            "action": action,
            "new_id_suggestion": new_id_suggestion,
            "embed_time_ms": float(embed_time_ms),
            "search_time_ms": float(search_time_ms),
            "total_time_ms": float(total_time_ms),
        }

        return report