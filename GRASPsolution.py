import numpy as np

from utils import renumerar_labels

def clave_par(a, b):
    """Clave ordenada para dict de pares (a<b)."""
    return (a, b) if a < b else (b, a)


def delta_modularidad_merge(M, S_a, W_in_a, S_b, W_in_b, W_ab):
    """
    Delta de Q al fusionar comunidades a y b, usando:
      Q_c = (W_in_c / M) - (S_c / (2M))^2
    donde:
      M = suma de pesos de aristas (cada arista una vez)
      S_c = suma de strengths de nodos en comunidad c
      W_in_c = suma de pesos de aristas internas (cada arista una vez)
      W_ab = suma de pesos de aristas entre a y b (cada arista una vez)
    """
    if M <= 0:
        return 0.0

    # Q antes
    qa = (W_in_a / M) - (S_a / (2.0 * M)) ** 2
    qb = (W_in_b / M) - (S_b / (2.0 * M)) ** 2

    # Q después (merge)
    S_new = S_a + S_b
    W_in_new = W_in_a + W_in_b + W_ab
    qnew = (W_in_new / M) - (S_new / (2.0 * M)) ** 2

    return qnew - (qa + qb)


def grasp_comunidades(
    prep,
    *,
    alpha=0.3,
    rcl_k=None,
    max_merges=None,
    semilla=None,
    permitir_merges_peores=False,
):
    """
    GRASP constructivo para CDP basado en fusiones (merges) de comunidades.

    - Empieza con cada nodo en su propia comunidad.
    - En cada paso evalúa posibles merges entre comunidades conectadas.
    - Construye una RCL (Restricted Candidate List) usando:
        * alpha (umbral por calidad) o
        * rcl_k (top-k)
    - Elige aleatoriamente un merge dentro de la RCL y lo aplica.
    - Devuelve labels (n,) con el id de comunidad por nodo.

    Parámetros:
      alpha: float en [0,1]. Más pequeño => más greedy. Más grande => más aleatorio.
      rcl_k: si no es None, usa top-k candidatos (ignora alpha).
      max_merges: límite de fusiones.
      semilla: para reproducibilidad.
      permitir_merges_peores: si False, para cuando mejor ΔQ <= 0.
    """
    rng = np.random.default_rng(semilla)

    n = prep["n"]
    s = prep["strength"]          # (n,)
    neighbors = prep["neighbors"] # lista de listas
    weights = prep["weights"]     # lista de dicts weights[i][j]=w

    # M = suma de pesos de aristas (cada arista una vez)
    M = float(s.sum() / 2.0)
    if M <= 0 or n == 0:
        return np.zeros(n, dtype=int)

    # Inicialización: cada nodo es su comundad
    comm_de_nodo = np.arange(n, dtype=int)

    # Estadísticos por comunidad (solo para comunidades activas)
    activa = np.ones(n, dtype=bool)
    S = s.astype(float).copy()               # S[c]
    W_in = np.zeros(n, dtype=float)          # W_in[c] (singletons => 0)

    # W_between[(a,b)] = suma de pesos entre comunidades a y b (a<b), solo si >0
    W_between = {}

    # Construimos W_between desde aristas (i<j)
    for i in range(n):
        ci = comm_de_nodo[i]
        for j in neighbors[i]:
            if j <= i:
                continue
            cj = comm_de_nodo[j]
            if ci == cj:
                continue
            key = clave_par(ci, cj)
            W_between[key] = W_between.get(key, 0.0) + float(weights[i][j])

    # Helper para listar candidatos actuales
    def listar_candidatos():
        # Devuelve lista de (a, b, deltaQ, W_ab) para pares activos con W_ab>0
        candidatos = []
        for (a, b), wab in W_between.items():
            if (not activa[a]) or (not activa[b]):
                continue
            dQ = delta_modularidad_merge(M, S[a], W_in[a], S[b], W_in[b], wab)
            candidatos.append((a, b, dQ, wab))
        return candidatos

    merges_hechos = 0
    while True:
        if max_merges is not None and merges_hechos >= max_merges:
            break

        candidatos = listar_candidatos()
        if not candidatos:
            break

        # Ordenamos por deltaQ (mejor primero)
        candidatos.sort(key=lambda x: x[2], reverse=True)
        mejor_dQ = candidatos[0][2]

        if (not permitir_merges_peores) and (mejor_dQ <= 0.0):
            break

        # Construcción RCL
        if rcl_k is not None:
            rcl = candidatos[: max(1, min(rcl_k, len(candidatos)))]
        else:
            # Umbral alpha basado en rango [mejor, peor]
            peor_dQ = candidatos[-1][2]
            umbral = mejor_dQ - alpha * (mejor_dQ - peor_dQ)
            rcl = [c for c in candidatos if c[2] >= umbral]
            if not rcl:
                rcl = [candidatos[0]]

        # Elegimos aleatoriamente dentro de la RCL (uniforme)
        a, b, dQ, wab = rcl[rng.integers(0, len(rcl))]

        # Aplicar merge: absorbemos b en a
        if b < a:
            a, b = b, a

        # Actualizamos stats de a
        W_in[a] = W_in[a] + W_in[b] + W_between.get(clave_par(a, b), 0.0)
        S[a] = S[a] + S[b]

        # Marcar b como inactiva
        activa[b] = False

        # Reasignar nodos que estaban en b hacia a
        comm_de_nodo[comm_de_nodo == b] = a

        # Actualizar W_between: combinar conexiones de b con otros en a
        claves_a_eliminar = []
        actualizaciones = {}  # key -> nuevo_w

        for (x, y), wxy in W_between.items():
            if x == b or y == b:
                # par que involucra b
                otro = y if x == b else x
                if not activa[otro] or otro == a:
                    claves_a_eliminar.append((x, y))
                    continue

                # Nuevo peso entre a y "otro" = W(a,otro) + W(b,otro)
                key_ao = clave_par(a, otro)
                w_ao = W_between.get(key_ao, 0.0)
                actualizaciones[key_ao] = w_ao + wxy

                claves_a_eliminar.append((x, y))

        # Eliminar todas las entradas que tocaban b y también (a,b) si existía
        for k in claves_a_eliminar:
            W_between.pop(k, None)
        W_between.pop(clave_par(a, b), None)

        # Aplicar actualizaciones
        for k, val in actualizaciones.items():
            if val > 0 and activa[k[0]] and activa[k[1]]:
                W_between[k] = val

        merges_hechos += 1

    # Normalizar labels a 0..K-1
    return renumerar_labels(comm_de_nodo)
