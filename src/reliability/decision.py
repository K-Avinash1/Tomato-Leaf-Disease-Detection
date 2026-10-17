import json

import numpy as np

from src.reliability.knn import KnnBank


def softmax_max(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=1, keepdims=True)
    return p.max(axis=1)


class ReliabilityGate:
    """unsupported: far from tomato training images | uncertain: low confidence | reliable: neither."""

    def __init__(self, bank, tau_knn, tau_conf):
        self.bank, self.tau_knn, self.tau_conf = bank, float(tau_knn), float(tau_conf)

    @classmethod
    def from_files(cls, config_path, bank_path):
        cfg = json.load(open(config_path))
        return cls(KnnBank.load(bank_path), cfg["tau_knn"], cfg["tau_conf"])

    def decide(self, logits, feats):
        sim = self.bank.similarity(feats)
        conf = softmax_max(logits)
        status = np.where(sim < self.tau_knn, "unsupported",
                          np.where(conf < self.tau_conf, "uncertain", "reliable"))
        return status, sim, conf