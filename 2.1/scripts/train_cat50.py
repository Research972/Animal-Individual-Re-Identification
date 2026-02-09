import argparse
from pathlib import Path

from catreid.config.default import get_default_cfg
from catreid.utils.device import get_device
from catreid.utils.seed import seed_everything
from catreid.transforms.reid_transforms import build_transforms
from catreid.datasets.folder_dataset import build_datasets_from_folder
from catreid.samplers.pk_sampler import PKSampler
from catreid.models.resnet_embedder import ResNet50Embedder
from catreid.trainers.trainer import Trainer


def parse_args():
    p = argparse.ArgumentParser("Train Cat ReID (PK + Triplet) (no faiss)")
    p.add_argument("--data_root", type=str, required=True,
                   help="Path to cat50 folder, e.g. /Users/guo/.../data_cat50_big/cat50")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--P", type=int, default=8, help="Identities per batch (P)")
    p.add_argument("--K", type=int, default=2, help="Images per identity (K)")
    p.add_argument("--steps_per_epoch", type=int, default=120)
    p.add_argument("--emb_dim", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--margin", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split_ratio", type=float, default=0.8)
    p.add_argument("--batch_eval", type=int, default=128)
    p.add_argument("--chunk_eval", type=int, default=512,
                   help="Chunk size for similarity computation to save memory")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--save_dir", type=str, default="checkpoints",
                   help="Where to save best checkpoint")
    p.add_argument("--exp_name", type=str, default="cat50_resnet50_pk_triplet",
                   help="Checkpoint prefix")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = get_default_cfg()
    cfg.update(vars(args))

    seed_everything(cfg["seed"])
    device = get_device(prefer_mps=True)
    print(f"Device: {device}")

    data_root = Path(cfg["data_root"])
    if not data_root.exists():
        raise FileNotFoundError(f"--data_root not found: {data_root}")

    train_tf, test_tf = build_transforms()

    train_set, test_set, num_ids = build_datasets_from_folder(
        data_root=str(data_root),
        split_ratio=cfg["split_ratio"],
        train_transform=train_tf,
        test_transform=test_tf,
        seed=cfg["seed"],
    )
    print(f"Identities: {num_ids}")
    print(f"Train images: {len(train_set)}  Test images: {len(test_set)}")

    if num_ids < cfg["P"]:
        raise RuntimeError(f"Not enough identities for P={cfg['P']}. Have {num_ids}.")

    sampler = PKSampler(
        indices_by_id=train_set.indices_by_id,
        P=cfg["P"],
        K=cfg["K"],
        steps_per_epoch=cfg["steps_per_epoch"],
        seed=cfg["seed"],
    )

    import torch
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_sampler=sampler,
        num_workers=cfg["num_workers"],
        pin_memory=False,
    )

    model = ResNet50Embedder(emb_dim=cfg["emb_dim"], pretrained=True).to(device)
    print("[DEBUG] model param device =", next(model.parameters()).device)

    trainer = Trainer(
        model=model,
        device=device,
        lr=cfg["lr"],
        margin=cfg["margin"],
        save_dir=cfg["save_dir"],
        exp_name=cfg["exp_name"],
        batch_eval=cfg["batch_eval"],
        chunk_eval=cfg["chunk_eval"],
        num_workers=cfg["num_workers"],
    )

    trainer.fit(
        train_loader=train_loader,
        train_set=train_set,
        test_set=test_set,
        epochs=cfg["epochs"],
    )


if __name__ == "__main__":
    main()
