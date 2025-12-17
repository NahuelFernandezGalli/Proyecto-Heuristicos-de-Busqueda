import numpy as np


def _init_stats(prep, labels):
    """
    Inicializa estadísticas para modularidad ponderada.

    Devuelve:
      - M: suma total de pesos de aristas (cada arista una vez)
      - s: strength por nodo
      - inv: labels comprimidos a 0..k-1 (por si labels no son 0..k-1)
      - comm_ids: valores originales únicos de labels
      - counts: tamaño de cada comunidad
      - S_c: suma de strengths por comunidad
      - W_in_c: suma de pesos internos por comunidad (cada arista una vez)
      - Q: modularidad actual
    """
    labels = np.asarray(labels)
    s = prep["strength"]
    neighbors = prep["neighbors"]
    weights = prep["weights"]
    n = prep["n"]

    M = s.sum() / 2.0
    if M == 0:
        # Grafo sin aristas
        comm_ids, inv = np.unique(labels, return_inverse=True)
        k = len(comm_ids)
        counts = np.bincount(inv, minlength=k)
        S_c = np.zeros(k, dtype=float)
        np.add.at(S_c, inv, s)
        W_in_c = np.zeros(k, dtype=float)
        Q = -np.sum((S_c / (2.0 * M + 1e-12)) ** 2)  # evita /0
        return M, s, inv, comm_ids, counts, S_c, W_in_c, float(Q)

    comm_ids, inv = np.unique(labels, return_inverse=True)
    k = len(comm_ids)

    counts = np.bincount(inv, minlength=k)

    S_c = np.zeros(k, dtype=float)
    np.add.at(S_c, inv, s)

    W_in_c = np.zeros(k, dtype=float)
    # suma interna por comunidad (cada arista una vez)
    for i in range(n):
        ci = inv[i]
        for j in neighbors[i]:
            if j <= i:
                continue
            if inv[j] == ci:
                W_in_c[ci] += weights[i][j]

    Q = np.sum((W_in_c / M) - (S_c / (2.0 * M)) ** 2)
    return M, s, inv, comm_ids, counts, S_c, W_in_c, float(Q)


def _deltaQ_move(prep, i, c_from, c_to, inv, s, M, S_c, W_in_c):
    """
    Calcula deltaQ al mover nodo i de comunidad c_from -> c_to,
    en O(deg(i)).

    Usa fórmula comunitaria:
      Q = sum_c [ W_in_c/M - (S_c/(2M))^2 ]
    """
    if c_from == c_to:
        return 0.0

    neighbors = prep["neighbors"]
    weights = prep["weights"]

    # Peso de i hacia nodos en c_from y hacia nodos en c_to
    w_to_from = 0.0
    w_to_to = 0.0
    for j in neighbors[i]:
        w = weights[i][j]
        cj = inv[j]
        if cj == c_from:
            w_to_from += w
        elif cj == c_to:
            w_to_to += w

    si = s[i]

    # Actualizaciones hipotéticas:
    # W_in_from' = W_in_from - w_to_from
    # W_in_to'   = W_in_to   + w_to_to
    # S_from' = S_from - si
    # S_to'   = S_to   + si

    Wf = W_in_c[c_from]
    Wt = W_in_c[c_to]
    Sf = S_c[c_from]
    St = S_c[c_to]

    Wf_new = Wf - w_to_from
    Wt_new = Wt + w_to_to
    Sf_new = Sf - si
    St_new = St + si

    # delta contribución comunidades afectadas (solo from y to)
    # Parte 1: interna
    d_in = (Wf_new - Wf) / M + (Wt_new - Wt) / M

    # Parte 2: término de grados
    # -( (S/(2M))^2 )  -> delta = -[(Sf_new/(2M))^2 - (Sf/(2M))^2] -[(St_new/(2M))^2 - (St/(2M))^2]
    denom = (2.0 * M)
    d_deg = -((Sf_new / denom) ** 2 - (Sf / denom) ** 2) - ((St_new / denom) ** 2 - (St / denom) ** 2)

    return float(d_in + d_deg), w_to_from, w_to_to


def simulated_annealing_modularity(
    prep,
    k,
    max_evals=100_000,
    T0=0.05,
    alpha=0.9995,
    seed=0,
    init_labels=None,
    enforce_nonempty=True,
    return_history=False,
):
    """
    SA para maximizar modularidad ponderada con EXACTAMENTE k comunidades (labels en 0..k-1).

    - Movimiento: elegir nodo i y cambiar su comunidad a otra.
    - Aceptación: si deltaQ >= 0 aceptar; si no, aceptar con prob exp(deltaQ / T).
      (deltaQ es negativo en movimientos peores)

    Parámetros:
      k: número de comunidades (2..100 según el enunciado)
      T0: temperatura inicial (en escala de modularidad suele funcionar algo como 0.01..0.2)
      alpha: enfriamiento geométrico T <- T*alpha
      enforce_nonempty: evita dejar una comunidad vacía (recomendado si quieres exactamente k)
    """
    rng = np.random.default_rng(seed)
    n = prep["n"]

    # Inicialización de labels
    if init_labels is None:
        labels = rng.integers(0, k, size=n, dtype=int)
        if enforce_nonempty:
            # asegurar que todas las comunidades aparecen al menos una vez
            # (reparación simple)
            missing = set(range(k)) - set(labels.tolist())
            missing = list(missing)
            if missing:
                # reasignamos algunos nodos aleatorios a las comunidades que faltan
                idxs = rng.choice(n, size=len(missing), replace=False)
                for node_idx, comm in zip(idxs, missing):
                    labels[node_idx] = comm
    else:
        labels = np.asarray(init_labels, dtype=int).copy()
        if labels.shape != (n,):
            raise ValueError(f"init_labels debe tener forma (n,), recibido {labels.shape}")
        if labels.min() < 0 or labels.max() >= k:
            raise ValueError("init_labels debe contener comunidades en 0..k-1")

    # Estadísticas iniciales (ojo: aquí labels ya es 0..k-1 => comm_ids=0..k-1)
    M, s, inv, comm_ids, counts, S_c, W_in_c, Q = _init_stats(prep, labels)

    best_labels = inv.copy()
    best_Q = Q

    T = float(T0)
    history = []
    eps = 1e-12

    for it in range(1, max_evals + 1):
        # enfriar
        T = max(T * alpha, eps)

        # proponer movimiento
        i = rng.integers(0, n)
        c_from = inv[i]

        # si no queremos vacíos: no permitimos sacar el último de su comunidad
        if enforce_nonempty and counts[c_from] <= 1:
            continue

        # elegir comunidad destino distinta
        c_to = rng.integers(0, k - 1)
        if c_to >= c_from:
            c_to += 1

        dQ, w_to_from, w_to_to = _deltaQ_move(prep, i, c_from, c_to, inv, s, M, S_c, W_in_c)

        # criterio de aceptación (maximización)
        if dQ >= 0:
            accept = True
        else:
            # prob = exp(dQ/T) (dQ<0 => prob en (0,1))
            accept = (rng.random() < np.exp(dQ / T))

        if accept:
            # aplicar cambio a estructuras
            inv[i] = c_to
            Q += dQ

            # actualizar counts
            counts[c_from] -= 1
            counts[c_to] += 1

            # actualizar S_c
            si = s[i]
            S_c[c_from] -= si
            S_c[c_to] += si

            # actualizar W_in_c
            W_in_c[c_from] -= w_to_from
            W_in_c[c_to] += w_to_to

            # actualizar mejor
            if Q > best_Q:
                best_Q = Q
                best_labels = inv.copy()

        if return_history and (it % 1000 == 0):
            history.append((it, Q, best_Q, T))

    if return_history:
        return best_labels, float(best_Q), history
    return best_labels, float(best_Q)
