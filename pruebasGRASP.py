import numpy as np

from GRASPsolution import grasp_comunidades
from funcionObjetivo import modularidad_ponderada
from experimentos import cargar_y_preparar, resumen_Q_K


def probar_grasp(
    semillas=(0, 1, 2, 3, 4),
    alpha=0.3,
    rcl_k=None,
    max_merges=None,
    permitir_merges_peores=False,
):
    _, prep = cargar_y_preparar()

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

    resumen_Q_K(Qs, Ks, titulo="\n=== RESUMEN ===")


if __name__ == "__main__":
    probar_grasp(
        semillas=(0, 1, 2, 3, 4),
        alpha=0.30,       # más chico => más greedy, más grande => más aleatorio
        rcl_k=None,       
        max_merges=None,  # limitar tiempo, número (ej: 5000)
        permitir_merges_peores=False,
    )
