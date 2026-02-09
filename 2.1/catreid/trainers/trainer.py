from dataclasses import dataclass
from pathlib import Path
import time

import torch
import torch.nn as nn
import torch.optim as optim

from catreid.metrics.retrieval import compute_rank1_map
from catreid.utils.io import ensure_dir, save_checkpoint


# -------------------------
# Triplet loss (batch-hard)
# -------------------------
def pairwise_d2(emb: torch.Tensor) -> torch.Tensor:
    # emb is L2-normalized => d^2 = 2 - 2*cos
    sim = emb @ emb.t()
    return (2.0 - 2.0 * sim).clamp_min(0.0)


def batch_hard_triplet_loss(emb: torch.Tensor, labels: torch.Tensor, margin: float = 0.2) -> torch.Tensor:
    d2 = pairwise_d2(emb)
    B = emb.size(0)
    labels = labels.view(-1, 1)
    same = (labels == labels.t())
    diff = ~same
    eye = torch.eye(B, device=emb.device, dtype=torch.bool)
    same = same & (~eye)

    pos = d2.masked_fill(~same, -1.0)
    hardest_pos, _ = pos.max(dim=1)

    neg = d2.masked_fill(~diff, 1e9)
    hardest_neg, _ = neg.min(dim=1)

    loss = torch.relu(hardest_pos - hardest_neg + margin)
    return loss.mean()


# -------------------------
# Embedding extraction
# -------------------------
@torch.no_grad()
def get_all_embeddings(dataset, model, device, batch_size=128, num_workers=0):
    model.eval()
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    embs, labs = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        z = model(x).detach().cpu()
        embs.append(z)
        labs.append(torch.as_tensor(y).detach().cpu())
    return torch.cat(embs, dim=0), torch.cat(labs, dim=0)


# -------------------------
# Trainer
# -------------------------
@dataclass
class Trainer:
    model: nn.Module
    device: torch.device
    lr: float = 3e-4
    margin: float = 0.2
    save_dir: str = "checkpoints"
    exp_name: str = "cat50_resnet50_pk_triplet"
    batch_eval: int = 128
    chunk_eval: int = 512
    num_workers: int = 0
    eval_on_cpu: bool = True  # ✅ 默认评估在 CPU 上（更稳）

    def __post_init__(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        ensure_dir(self.save_dir)
        self.best_map = -1.0
        self.best_path = str(Path(self.save_dir) / f"best_{self.exp_name}.pt")
        print(f"Saving best checkpoint to: {self.best_path}")

    def train_one_epoch(self, train_loader, epoch: int, log_every: int = 20):
        self.model.train()
        loss_sum = 0.0

        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(self.device, non_blocking=True)
            y = torch.as_tensor(y, device=self.device)

            if batch_idx == 0:
                print("[DEBUG] batch x device:", x.device, "y device:", y.device)

            self.optimizer.zero_grad(set_to_none=True)
            emb = self.model(x)
            loss = batch_hard_triplet_loss(emb, y, margin=self.margin)
            loss.backward()
            self.optimizer.step()

            loss_sum += float(loss.item())
            if batch_idx % log_every == 0:
                print(f"Epoch {epoch:03d} Iter {batch_idx:03d} | loss={loss.item():.4f}")

        avg_loss = loss_sum / max(1, len(train_loader))
        return avg_loss

    @torch.no_grad()
    def evaluate(self, train_set, test_set):
        """
        Return dict: {"rank1": float, "mAP": float}
        """
        # 1) embeddings 提取：可以走 MPS（更快）
        gallery_emb, gallery_lab = get_all_embeddings(
            train_set, self.model, self.device,
            batch_size=self.batch_eval, num_workers=self.num_workers
        )
        query_emb, query_lab = get_all_embeddings(
            test_set, self.model, self.device,
            batch_size=self.batch_eval, num_workers=self.num_workers
        )

        # 2) 指标计算：默认 CPU（更稳，避免 MPS 大矩阵卡/慢/炸显存）
        if self.eval_on_cpu:
            gallery_emb = gallery_emb.cpu()
            gallery_lab = gallery_lab.cpu()
            query_emb = query_emb.cpu()
            query_lab = query_lab.cpu()

        # 你 metrics/retrieval.py 里 compute_rank1_map 需要返回 (rank1, mAP)
        rank1, mAP = compute_rank1_map(
            query_emb, query_lab,
            gallery_emb, gallery_lab,
            chunk=self.chunk_eval,   # 如果你函数里参数名叫 chunk_size，请改成 chunk_size=
        )

        return {"rank1": float(rank1), "mAP": float(mAP)}

    def maybe_save_best(self, mAP: float, epoch: int):
        if mAP > self.best_map:
            self.best_map = mAP
            meta = {"best_epoch": int(epoch), "best_mAP": float(mAP)}
            save_checkpoint(self.best_path, self.model.state_dict(), meta)
            print(f"[OK] Saved BEST: {self.best_path}  (mAP={mAP:.4f})")

    def fit(self, train_loader, train_set, test_set, epochs: int):
        for ep in range(1, epochs + 1):
            epoch_start = time.time()

            # ---- train ----
            train_start = time.time()
            avg_loss = self.train_one_epoch(train_loader, ep)
            train_time = time.time() - train_start

            # ---- eval ----
            eval_start = time.time()
            metrics = self.evaluate(train_set, test_set)
            eval_time = time.time() - eval_start

            epoch_time = time.time() - epoch_start

            print(
                f"[Epoch {ep:03d}] "
                f"loss={avg_loss:.4f} | "
                f"R1={metrics['rank1']:.4f} | "
                f"mAP={metrics['mAP']:.4f} | "
                f"train={train_time/60:.1f}min | "
                f"eval={eval_time/60:.1f}min | "
                f"total={epoch_time/60:.1f}min"
            )

            # ---- save best ----
            self.maybe_save_best(metrics["mAP"], ep)