import numpy as np


def softmax_np(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def nll_and_ece(logits, labels, n_bins=15):
    """Negative log-likelihood and expected calibration error (15 equal-width confidence bins)."""
    p = softmax_np(np.asarray(logits, dtype=np.float64))
    labels = np.asarray(labels)
    n = len(labels)
    nll = float(-np.log(np.clip(p[np.arange(n), labels], 1e-12, None)).mean())
    conf = p.max(axis=1)
    correct = (p.argmax(axis=1) == labels).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return nll, float(ece)