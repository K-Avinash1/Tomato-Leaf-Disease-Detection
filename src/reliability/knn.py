import numpy as np


class KnnBank:
    """Cosine similarity to the k-th most similar tomato TRAINING image."""

    def __init__(self, bank, k=10):
        self.bank = bank.astype(np.float32)          # unit-length rows
        self.k = k

    @classmethod
    def from_features(cls, feats, k=10):
        return cls(feats / np.linalg.norm(feats, axis=1, keepdims=True), k)

    def similarity(self, feats):
        q = (feats / np.linalg.norm(feats, axis=1, keepdims=True)).astype(np.float32)
        out = np.empty(len(q), dtype=np.float32)
        for s in range(0, len(q), 512):
            sims = q[s:s + 512] @ self.bank.T
            out[s:s + 512] = np.partition(sims, -self.k, axis=1)[:, -self.k]
        return out

    def save(self, path):
        np.savez_compressed(path, bank=self.bank.astype(np.float16), k=self.k)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        return cls(z["bank"].astype(np.float32), int(z["k"]))