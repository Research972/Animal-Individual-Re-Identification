import random
import torch

class PKSampler(torch.utils.data.Sampler):
    """
    Each batch: P identities, K images per identity => batch size = P*K
    """
    def __init__(self, indices_by_id, P=8, K=2, steps_per_epoch=120, seed=42):
        self.indices_by_id = indices_by_id
        self.ids = sorted(indices_by_id.keys())
        self.P, self.K = P, K
        self.steps = steps_per_epoch
        self.rng = random.Random(seed)
        if len(self.ids) < P:
            raise ValueError(f"Need at least P identities. Have {len(self.ids)}, P={P}")

    def __len__(self):
        return self.steps

    def __iter__(self):
        for _ in range(self.steps):
            chosen_ids = self.rng.sample(self.ids, self.P)
            batch = []
            for cid in chosen_ids:
                pool = self.indices_by_id[cid]
                if len(pool) >= self.K:
                    pick = self.rng.sample(pool, self.K)
                else:
                    pick = [self.rng.choice(pool) for _ in range(self.K)]
                batch.extend(pick)
            yield batch
