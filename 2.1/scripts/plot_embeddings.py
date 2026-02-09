import argparse
from pathlib import Path
import os

import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from catreid.utils.device import get_device
from catreid.transforms.reid_transforms import build_transforms
from catreid.datasets.folder_dataset import build_datasets_from_folder
from catreid.models.resnet_embedder import ResNet50Embedder


@torch.no_grad()
def get_all_embeddings(dataset, model, device, batch_size=128, num_workers=0, max_points=None):
    model.eval()
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    embs, labs = [], []
    seen = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        z = model(x).detach().cpu()
        y = torch.as_tensor(y).detach().cpu()
        embs.append(z)
        labs.append(y)
        seen += z.size(0)
        if max_points is not None and seen >= max_points:
            break

    E = torch.cat(embs, dim=0)
    Y = torch.cat(labs, dim=0)
    if max_points is not None and E.size(0) > max_points:
        E = E[:max_points]
        Y = Y[:max_points]
    return E, Y


def scatter_plot_2d(Z2, labels, title, out_path, max_classes=50):
    plt.figure(figsize=(9, 7))
    labels = labels.astype(int)

    # 为了不让 legend 爆炸：只给每个类一个点的标签
    shown = set()
    for i in range(Z2.shape[0]):
        c = labels[i]
        if c >= max_classes:
            continue
        if c not in shown:
            plt.scatter(Z2[i, 0], Z2[i, 1], s=10, label=f"id_{c:03d}")
            shown.add(c)
        else:
            plt.scatter(Z2[i, 0], Z2[i, 1], s=10)

    plt.title(title)
    plt.xlabel("dim-1")
    plt.ylabel("dim-2")
    # legend 太多会很挤，50 类也还行；不想要就注释掉
    plt.legend(markerscale=2, fontsize=7, ncol=2, frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def parse_args():
    p = argparse.ArgumentParser("Plot embedding distribution (PCA/t-SNE)")
    p.add_argument("--data_root", type=str, required=True, help="path to cat50 folder")
    p.add_argument("--ckpt", type=str, required=True, help="best checkpoint .pt path")
    p.add_argument("--split_ratio", type=float, default=0.8)
    p.add_argument("--emb_dim", type=int, default=256)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--max_points", type=int, default=2000, help="cap points for tsne speed")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", type=str, default="outputs")
    p.add_argument("--which", type=str, default="test", choices=["train", "test", "all"])
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = get_device(prefer_mps=True)
    print("Device:", device)

    data_root = Path(args.data_root)
    ckpt_path = Path(args.ckpt)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_tf, test_tf = build_transforms()

    train_set, test_set, num_ids = build_datasets_from_folder(
        data_root=str(data_root),
        split_ratio=args.split_ratio,
        train_transform=train_tf,
        test_transform=test_tf,
        seed=args.seed,
    )
    print(f"Identities: {num_ids} | train={len(train_set)} test={len(test_set)}")

    model = ResNet50Embedder(emb_dim=args.emb_dim, pretrained=False).to(device)

    # 兼容你 save_checkpoint 里只保存 state_dict 的情况
    state = torch.load(str(ckpt_path), map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        # 如果你曾经保存过 {"model": state_dict, ...}
        model.load_state_dict(state["model"], strict=True)
    elif isinstance(state, dict) and all(isinstance(k, str) for k in state.keys()):
        # save_checkpoint(model_state_dict) 这种
        model.load_state_dict(state, strict=True)
    else:
        raise RuntimeError(f"Unrecognized checkpoint format: {type(state)}")

    # 选择画哪个集合
    if args.which == "train":
        ds_list = [("train", train_set)]
    elif args.which == "test":
        ds_list = [("test", test_set)]
    else:
        ds_list = [("train", train_set), ("test", test_set)]

    for name, ds in ds_list:
        E, Y = get_all_embeddings(
            ds, model, device,
            batch_size=args.batch,
            num_workers=args.num_workers,
            max_points=args.max_points,
        )
        E = E.numpy().astype(np.float32)
        Y = Y.numpy().astype(np.int64)
        print(f"[{name}] extracted: {E.shape[0]} embeddings")

        # ---- PCA ----
        pca = PCA(n_components=2, random_state=args.seed)
        Zp = pca.fit_transform(E)
        scatter_plot_2d(
            Zp, Y,
            title=f"PCA 2D ({name})",
            out_path=str(out_dir / f"emb_{name}_pca.png"),
        )

        # ---- t-SNE ----
        # t-SNE 非常吃时间，max_points 建议 <= 2000
        tsne = TSNE(
            n_components=2,
            perplexity=30,
            learning_rate="auto",
            init="pca",
            random_state=args.seed,
        )
        Zt = tsne.fit_transform(E)
        scatter_plot_2d(
            Zt, Y,
            title=f"t-SNE 2D ({name})",
            out_path=str(out_dir / f"emb_{name}_tsne.png"),
        )

        print(f"[OK] saved: {out_dir}/emb_{name}_pca.png and emb_{name}_tsne.png")

    print("Done.")


if __name__ == "__main__":
    main()