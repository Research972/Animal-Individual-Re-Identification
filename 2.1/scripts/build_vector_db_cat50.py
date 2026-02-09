# scripts/build_vector_db_cat50.py
import argparse
from pathlib import Path
import time

import torch

from catreid.utils.device import get_device
from catreid.utils.seed import seed_everything
from catreid.transforms.reid_transforms import build_transforms
from catreid.datasets.folder_dataset import build_datasets_from_folder
from catreid.models.resnet_embedder import ResNet50Embedder
from catreid.utils.io import ensure_dir


def parse_args():
    p = argparse.ArgumentParser("Build vector DB for Cat ReID (gallery embeddings)")
    p.add_argument("--ckpt", type=str, required=True, help="Path to best_*.pt checkpoint")
    p.add_argument("--data_root", type=str, required=True, help="Path to catXX folder with id_ folders")
    p.add_argument("--out", type=str, default="checkpoints/vector_db_cat50.pt", help="Output vector db path")
    p.add_argument("--split_ratio", type=float, default=0.8, help="Same split_ratio as training")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--emb_dim", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--use_cpu", action="store_true", help="Force CPU for embedding extraction")
    return p.parse_args()


@torch.no_grad()
def extract_embeddings(dataset, model, device, batch_size=128, num_workers=0):
    model.eval()
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    embs, labs, paths = [], [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        z = model(x).detach().cpu()
        embs.append(z)
        labs.append(torch.as_tensor(y).detach().cpu())

    embs = torch.cat(embs, dim=0)  # [N, D]
    labs = torch.cat(labs, dim=0)  # [N]
    return embs, labs


def load_ckpt_state_dict(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    # 兼容两种常见保存格式：
    # 1) {"model": state_dict, ...}
    # 2) 直接就是 state_dict
    if isinstance(ckpt, dict) and ("model" in ckpt) and isinstance(ckpt["model"], dict):
        return ckpt["model"], ckpt
    if isinstance(ckpt, dict):
        # 可能直接是 state_dict
        return ckpt, {"raw": True}
    raise RuntimeError("Unknown checkpoint format")


def main():
    args = parse_args()
    seed_everything(args.seed)

    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"--data_root not found: {data_root}")

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"--ckpt not found: {ckpt_path}")

    out_path = Path(args.out)
    ensure_dir(str(out_path.parent))

    device = torch.device("cpu") if args.use_cpu else get_device(prefer_mps=True)
    print(f"Device: {device}")

    # --- build datasets (must match training split!) ---
    train_tf, test_tf = build_transforms()
    train_set, test_set, num_ids = build_datasets_from_folder(
        data_root=str(data_root),
        split_ratio=args.split_ratio,
        train_transform=train_tf,
        test_transform=test_tf,
        seed=args.seed,
    )
    print(f"Identities: {num_ids}")
    print(f"Gallery(train) images: {len(train_set)} | Query(test) images: {len(test_set)}")

    # --- build model ---
    model = ResNet50Embedder(emb_dim=args.emb_dim, pretrained=False).to(device)

    state_dict, meta = load_ckpt_state_dict(str(ckpt_path))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print("[WARN] Missing keys:", missing)
    if unexpected:
        print("[WARN] Unexpected keys:", unexpected)

    # debug: confirm model on device
    p0 = next(model.parameters())
    print(f"[DEBUG] model param device = {p0.device}")

    # --- extract gallery embeddings ---
    t0 = time.time()
    gallery_emb, gallery_lab = extract_embeddings(
        train_set, model, device,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    dt = time.time() - t0

    print(f"[OK] Extracted gallery embeddings: {gallery_emb.shape} in {dt/60:.2f} min")

    # --- save vector db ---
    # 注意：这里保存 label->id 的映射信息（便于推理输出 human-readable）
    # 你当前数据是 id_000... 这种结构，所以 label 就是 0..N-1
    db = {
        "embeddings": gallery_emb.contiguous(),  # [N, D], CPU tensor
        "labels": gallery_lab.contiguous(),      # [N]
        "ids": gallery_lab.contiguous(),
        "num_ids": int(num_ids),
        "emb_dim": int(gallery_emb.shape[1]),
        "split_ratio": float(args.split_ratio),
        "seed": int(args.seed),
        "data_root": str(data_root),
        "ckpt": str(ckpt_path),
    }
    torch.save(db, str(out_path))
    print(f"[OK] Saved vector DB to: {out_path}")


if __name__ == "__main__":
    main()