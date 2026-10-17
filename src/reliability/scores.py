import numpy as np
from scipy.special import logsumexp
from sklearn.covariance import LedoitWolf

K_NN = 10


class Scorer:
    """Fitted on tomato TRAINING images only. Every score: higher = more like a normal tomato image."""

    def __init__(self, train_feats, train_labels, n_classes=10):
        X = train_feats.astype(np.float64)
        self.means = np.stack([X[train_labels == c].mean(0) for c in range(n_classes)])
        self.precision = LedoitWolf().fit(X - self.means[train_labels]).precision_
        self.bank = (train_feats / np.linalg.norm(train_feats, axis=1, keepdims=True)).astype(np.float32)

    def mahalanobis_min(self, feats):
        X = feats.astype(np.float64)
        d = np.empty((len(X), len(self.means)))
        for c, mu in enumerate(self.means):
            diff = X - mu
            d[:, c] = np.einsum("nd,nd->n", diff @ self.precision, diff)
        return d.min(axis=1)

    def knn_similarity(self, feats, k=K_NN):
        Q = (feats / np.linalg.norm(feats, axis=1, keepdims=True)).astype(np.float32)
        out = np.empty(len(Q), dtype=np.float32)
        for s in range(0, len(Q), 512):
            sims = Q[s:s + 512] @ self.bank.T
            out[s:s + 512] = np.partition(sims, -k, axis=1)[:, -k]
        return out

    def all_scores(self, logits, feats):
        z = logits - logits.max(1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(1, keepdims=True)
        return {"max_softmax": p.max(1), "max_logit": logits.max(1),
                "energy": logsumexp(logits, axis=1),
                "mahalanobis": -self.mahalanobis_min(feats),
                "knn_cosine": self.knn_similarity(feats)}