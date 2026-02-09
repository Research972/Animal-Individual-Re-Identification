def get_default_cfg():
    # You can expand later. For now keep minimal.
    return {
        "epochs": 60,
        "P": 8,
        "K": 2,
        "steps_per_epoch": 120,
        "emb_dim": 256,
        "lr": 3e-4,
        "margin": 0.2,
        "seed": 42,
        "split_ratio": 0.8,
        "batch_eval": 128,
        "chunk_eval": 512,
        "num_workers": 0,
        "save_dir": "checkpoints",
        "exp_name": "cat50_resnet50_pk_triplet",
    }
