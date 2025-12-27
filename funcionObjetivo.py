import numpy as np

def modularidad_ponderada(prep, labels):
    """
    Modularity Q (ponderada) para una partición dada por labels.

    prep: salida de preparar_grafo(G)
      - strength: array s_i
      - weights: lista de dicts weights[i][j] = w_ij
      - neighbors: lista de listas con vecinos (índices)
      - n: número de nodos

    labels: array/list (n,) con la comunidad de cada nodo (0..k-1 o cualquier int)
    """
    labels = np.asarray(labels)
    s = prep["strength"]
    M = prep.get("m")
    if M is None:
        M = s.sum() / 2.0
    neighbors = prep["neighbors"]
    weights = prep["weights"]
    n = prep["n"]

    if M == 0:
        return 0.0

    # 1) S_c = suma de strengths por comunidad
    #    (equivalente a d_c en modularidad clásica)
    comm_ids, inv = np.unique(labels, return_inverse=True)
    k = len(comm_ids)
    S_c = np.zeros(k, dtype=float)
    np.add.at(S_c, inv, s)

    # 2) W_in_c = suma de pesos de aristas internas por comunidad (cada arista una vez)
    W_in_c = np.zeros(k, dtype=float)

    # Recorremos aristas como (i, j) con i<j usando neighbors/weights
    for i in range(n):
        ci = inv[i]
        for j in neighbors[i]:
            if j <= i:
                continue
            if inv[j] == ci:
                W_in_c[ci] += weights[i][j]

    # Fórmula: Q = sum_c [ (W_in_c / M) - (S_c / (2M))^2 ]
    Q = np.sum((W_in_c / M) - (S_c / (2.0 * M))**2)
    return float(Q)
