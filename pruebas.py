import numpy as np

from grafo import crear_grafo, preparar_grafo
from GRASPsolution import grasp_comunidades
from funcionObjetivo import modularidad_ponderada


def probar_grasp(
    semillas=(0, 1, 2, 3, 4),
    alpha=0.3,
    rcl_k=None,
    max_merges=None,
    permitir_merges_peores=False,
):
    print("Cargando grafo")
    G = crear_grafo()
    prep = preparar_grafo(G)

    print("Nodos:", prep["n"])
    print("Aristas:", G.number_of_edges())
    print()

    Qs = []
    Ks = []

    for seed in semillas:
        labels = grasp_comunidades(
            prep,
            alpha=alpha,
            rcl_k=rcl_k,
            max_merges=max_merges,
            semilla=seed,
            permitir_merges_peores=permitir_merges_peores,
        )

        Q = modularidad_ponderada(prep, labels)
        K = len(np.unique(labels))

        Qs.append(Q)
        Ks.append(K)

        print(f"GRASP seed = {seed:>3} | Q={Q:.6f} | K={K}")

    Qs = np.array(Qs, dtype=float)
    Ks = np.array(Ks, dtype=int)

    print("\n=== RESUMEN ===")
    print(f"Q: mejor = {Qs.max():.6f} | media={Qs.mean():.6f} | std={Qs.std():.6f} | peor={Qs.min():.6f}")
    print(f"K: mejor(Q) = {Ks[Qs.argmax()]} | media={Ks.mean():.2f} | std={Ks.std():.2f} | min={Ks.min()} | max={Ks.max()}")


if __name__ == "__main__":
    probar_grasp(
        semillas=(0, 1, 2, 3, 4),
        alpha=0.30,       # más chico => más greedy, más grande => más aleatorio
        rcl_k=None,       
        max_merges=None,  # limitar tiempo, número (ej: 5000)
        permitir_merges_peores=False,
    )
