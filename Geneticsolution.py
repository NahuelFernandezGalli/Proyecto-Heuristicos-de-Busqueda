import numpy as np

from utils import renumerar_labels


def inicial_aleatoria(prep, rng, p_nueva=0.02):
    """
    Inicialización aleatoria 'sensata':
    - Recorre nodos y los asigna a comunidad de un vecino con cierta probabilidad,
      o crea una nueva comunidad con probabilidad pequeña.
    """
    n = prep["n"]
    neighbors = prep["neighbors"]

    labels = np.full(n, -1, dtype=int)
    next_comm = 0

    for i in range(n):
        if labels[i] != -1:
            continue

        if rng.random() < p_nueva or len(neighbors[i]) == 0:
            labels[i] = next_comm
            next_comm += 1
            continue

        vecs = neighbors[i].copy()
        rng.shuffle(vecs)
        asignado = False
        for j in vecs:
            if labels[j] != -1:
                labels[i] = labels[j]
                asignado = True
                break
        if not asignado:
            labels[i] = next_comm
            next_comm += 1

    return renumerar_labels(labels)


def seleccion_torneo(fitness, rng, k=2):
    """Devuelve índice del ganador de torneo de tamaño k (maximiza fitness)."""
    n = len(fitness)
    idxs = rng.integers(0, n, size=k)
    best = idxs[0]
    for t in idxs[1:]:
        if fitness[t] > fitness[best]:
            best = t
    return best


def subconjunto_bfs(prep, rng, tam_objetivo):
    """
    Genera un subconjunto S de nodos usando BFS aleatorio desde una semilla,
    para que el cruce copie un 'bloque' coherente.
    """
    n = prep["n"]
    neighbors = prep["neighbors"]

    start = int(rng.integers(0, n))
    S = set([start])
    frontera = [start]

    while frontera and len(S) < tam_objetivo:
        u = frontera.pop(0)
        vecs = neighbors[u]
        if len(vecs) == 0:
            continue
        vecs = vecs.copy()
        rng.shuffle(vecs)
        for v in vecs:
            if v not in S:
                S.add(v)
                frontera.append(v)
            if len(S) >= tam_objetivo:
                break

    return np.fromiter(S, dtype=int)


def cruce_bloque(prep, padre_a, padre_b, rng, frac=0.3):
    """
    Cruce por bloque:
    - elige subconjunto S (BFS) del tamaño frac*n aprox
    - hijo toma labels de S desde A y el resto desde B
    """
    n = prep["n"]
    tam = max(1, int(frac * n))
    S = subconjunto_bfs(prep, rng, tam_objetivo=tam)

    hijo = np.array(padre_b, dtype=int, copy=True)
    hijo[S] = np.asarray(padre_a, dtype=int)[S]
    return renumerar_labels(hijo)


def mutacion_move_vecino_informada(prep, labels, rng, modularidad_ponderada, k_candidatos=10):
    """
    Mutación informada (best-improving):
    - elige un nodo i al azar
    - considera comunidades de hasta k_candidatos vecinos (muestreados si hay muchos)
    - evalúa el fitness y se queda con el mejor movimiento

    Nota: es más costosa que la mutación ciega, pero nos mejora mucho Q.
    """
    n = prep["n"]
    neighbors = prep["neighbors"]

    i = int(rng.integers(0, n))
    vecs = neighbors[i]
    if len(vecs) == 0:
        return labels

    # comunidades candidatas: las comunidades de algunos vecinos
    if len(vecs) > k_candidatos:
        idxs = rng.choice(len(vecs), size=k_candidatos, replace=False)
        cand_vecs = [vecs[t] for t in idxs]
    else:
        cand_vecs = vecs

    labels = np.asarray(labels, dtype=int)
    base_Q = modularidad_ponderada(prep, labels)

    mejor_labels = labels
    mejor_Q = base_Q

    # probar mover i a la comunidad del vecino j
    for j in cand_vecs:
        c_new = labels[j]
        if c_new == labels[i]:
            continue

        trial = np.array(labels, copy=True)
        trial[i] = c_new
        trial = renumerar_labels(trial)

        Q = modularidad_ponderada(prep, trial)
        if Q > mejor_Q:
            mejor_Q = Q
            mejor_labels = trial

    return mejor_labels


def genetico_comunidades(
    prep,
    modularidad_ponderada,
    *,
    tam_poblacion=60,
    generaciones=80,
    p_cruce=0.9,
    p_mutacion=0.3,
    elitismo=2,
    torneo_k=2,
    frac_cruce=0.3,
    p_inmigracion=0.03,
    k_candidatos_mut=10,
    semilla=None,
):
    """
    Algoritmo genético para CDP (maximiza modularidad ponderada).

    Mejoras incluidas:
      (2) Mutación informada: prueba varios movimientos y elige el mejor
      (3) Selección más suave (torneo_k=2) + elitismo controlado + inmigración

    Salida:
      - best_labels: labels del mejor individuo
      - best_Q: fitness (modularidad) del mejor individuo
      - historial: lista de (best_Q_generacion, avg_Q_generacion)
    """
    rng = np.random.default_rng(semilla)

    # Inicialización de población
    poblacion = [inicial_aleatoria(prep, rng) for _ in range(tam_poblacion)]
    fitness = np.array([modularidad_ponderada(prep, ind) for ind in poblacion], dtype=float)

    historial = []
    for gen in range(generaciones):
        # Orden para elitismo
        orden = np.argsort(-fitness)
        poblacion = [poblacion[i] for i in orden]
        fitness = fitness[orden]

        best_Q = float(fitness[0])
        avg_Q = float(fitness.mean())
        historial.append((best_Q, avg_Q))

        nueva = []

        # Elitismo
        for e in range(min(elitismo, tam_poblacion)):
            nueva.append(np.array(poblacion[e], copy=True))

        # Inmigración: mete algunos individuos nuevos aleatorios para mantener diversidad
        n_inm = int(round(p_inmigracion * tam_poblacion))
        for _ in range(n_inm):
            nueva.append(inicial_aleatoria(prep, rng))

        # Rellenar con hijos
        while len(nueva) < tam_poblacion:
            ia = seleccion_torneo(fitness, rng, k=torneo_k)
            ib = seleccion_torneo(fitness, rng, k=torneo_k)
            padre_a = poblacion[ia]
            padre_b = poblacion[ib]

            # Cruce
            if rng.random() < p_cruce:
                hijo = cruce_bloque(prep, padre_a, padre_b, rng, frac=frac_cruce)
            else:
                hijo = np.array(padre_a, dtype=int, copy=True)

            # Mutación (informada)
            if rng.random() < p_mutacion:
                hijo = mutacion_move_vecino_informada(
                    prep, hijo, rng,
                    modularidad_ponderada,
                    k_candidatos=k_candidatos_mut
                )

            nueva.append(hijo)

        poblacion = nueva
        fitness = np.array([modularidad_ponderada(prep, ind) for ind in poblacion], dtype=float)

    # Mejor final
    best_idx = int(np.argmax(fitness))
    best_labels = poblacion[best_idx]
    best_Q = float(fitness[best_idx])

    return best_labels, best_Q, historial
