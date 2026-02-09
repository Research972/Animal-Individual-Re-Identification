# catreid/metrics/retrieval.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import torch


@torch.no_grad()
def retrieval_metrics(
    query_emb: torch.Tensor,
    query_labels: torch.Tensor,
    gallery_emb: torch.Tensor,
    gallery_labels: torch.Tensor,
    chunk_size: int = 512,
) -> dict:
    """
    Compute Rank-1 and mAP for ReID (no faiss, pure torch, CPU-friendly).

    Args:
        query_emb:      [Q, D]  (usually L2-normalized embeddings)
        query_labels:   [Q]
        gallery_emb:    [G, D]
        gallery_labels: [G]
        chunk_size:     compute distances by chunks to save memory

    Returns:
        {"rank1": float, "mAP": float}
    """
    # Ensure CPU for stable/low-memory eval (trainer can already pass CPU tensors)
    query_emb = query_emb.detach().cpu()
    gallery_emb = gallery_emb.detach().cpu()
    query_labels = torch.as_tensor(query_labels).detach().cpu()
    gallery_labels = torch.as_tensor(gallery_labels).detach().cpu()

    # Ensure shapes [Q], [G]
    if query_labels.ndim > 1:
        query_labels = query_labels.view(-1)
    if gallery_labels.ndim > 1:
        gallery_labels = gallery_labels.view(-1)

    Q = query_emb.size(0)
    if Q == 0:
        return {"rank1": 0.0, "mAP": 0.0}

    rank1 = 0
    ap_sum = 0.0

    # Assumption: embeddings are L2-normalized
    # L2 distance^2 = 2 - 2*cosine_sim
    for i in range(Q):
        q = query_emb[i : i + 1]  # [1, D]
        y = query_labels[i]

        # compute distances in chunks
        dists = []
        G = gallery_emb.size(0)
        for j in range(0, G, chunk_size):
            g = gallery_emb[j : j + chunk_size]  # [c, D]
            dist = 2.0 - 2.0 * (q @ g.t())        # [1, c]
            dists.append(dist.squeeze(0))         # [c]

        dists = torch.cat(dists, dim=0)           # [G]
        order = torch.argsort(dists)              # asc: nearest first

        matches = (gallery_labels[order] == y).to(torch.int32)  # [G]

        # Rank-1
        if matches.numel() > 0 and matches[0].item() == 1:
            rank1 += 1

        # AP
        pos_cnt = int(matches.sum().item())
        if pos_cnt > 0:
            cumsum = torch.cumsum(matches, dim=0)                  # [G]
            ranks = torch.arange(1, matches.numel() + 1)           # [G]
            precision = (cumsum / ranks) * matches                 # only at positives
            ap = float(precision.sum().item()) / float(pos_cnt)
            ap_sum += ap

    return {
        "rank1": float(rank1) / float(Q),
        "mAP": float(ap_sum) / float(Q),
    }


@torch.no_grad()
def compute_rank1_map(
    query_emb: torch.Tensor,
    query_labels: torch.Tensor,
    gallery_emb: torch.Tensor,
    gallery_labels: torch.Tensor,
    chunk: int = 512,
):
    """
    Wrapper for Trainer: returns (rank1, mAP) as tuple.

    Trainer expects:
        rank1, mAP = compute_rank1_map(..., chunk=cfg["chunk_eval"])
    """
    m = retrieval_metrics(
        query_emb=query_emb,
        query_labels=query_labels,
        gallery_emb=gallery_emb,
        gallery_labels=gallery_labels,
        chunk_size=chunk,
    )
    return m["rank1"], m["mAP"]