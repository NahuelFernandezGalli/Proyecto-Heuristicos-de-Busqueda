import numpy as np


def renumerar_labels(labels):
    """Normaliza labels para que sean 0..K-1."""
    labels = np.asarray(labels, dtype=int)
    ids = np.unique(labels)
    remap = {old: new for new, old in enumerate(ids)}
    return np.array([remap[x] for x in labels], dtype=int)
