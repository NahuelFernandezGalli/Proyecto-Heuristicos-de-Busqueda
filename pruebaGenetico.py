import numpy as np

from funcionObjetivo import modularidad_ponderada
from Geneticsolution import genetico_comunidades
from experimentos import cargar_y_preparar, resumen_Q_K


def probar_genetico(
    semillas=(0, 1, 2, 3, 4),
    tam_poblacion=60,
    generaciones=80,
    p_cruce=0.9,
    p_mutacion=0.3,
    elitismo=2,
    torneo_k=2,
    frac_cruce=0.3,
    p_inmigracion=0.03,
    k_candidatos_mut=10,
):
    """
    Prueba el algoritmo genético (versión mejorada):
      - mutación informada (k_candidatos_mut)
      - selección más suave (torneo_k=2 por defecto)
      - inmigración para diversidad (p_inmigracion)
    """
    _, prep = cargar_y_preparar()

    Qs = []
    Ks = []

    for seed in semillas:
        labels, Q, hist = genetico_comunidades(
            prep,
            modularidad_ponderada,
            tam_poblacion=tam_poblacion,
            generaciones=generaciones,
            p_cruce=p_cruce,
            p_mutacion=p_mutacion,
            elitismo=elitismo,
            torneo_k=torneo_k,
            frac_cruce=frac_cruce,
            p_inmigracion=p_inmigracion,
            k_candidatos_mut=k_candidatos_mut,
            semilla=seed,
        )

        K = len(np.unique(labels))
        Qs.append(Q)
        Ks.append(K)

        best_last, avg_last = hist[-1]
        print(
            f"GA seed={seed:>3} | Q={Q:.6f} | K={K} | "
            f"last(best,avg)={best_last:.6f},{avg_last:.6f}"
        )

    resumen_Q_K(Qs, Ks, titulo="RESUMEN")


if __name__ == "__main__":
    probar_genetico(
        semillas=(0, 1, 2, 3, 4),
        tam_poblacion=60,
        generaciones=80,
        p_cruce=0.9,
        p_mutacion=0.3,
        elitismo=2,
        torneo_k=2,
        frac_cruce=0.3,
        p_inmigracion=0.03,
        k_candidatos_mut=10,
    )
