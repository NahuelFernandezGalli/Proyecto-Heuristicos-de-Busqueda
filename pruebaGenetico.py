import os
import json
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from funcionObjetivo import modularidad_ponderada
from Geneticsolution import genetico_comunidades
from experimentos import cargar_y_preparar


def probar_genetico(
    semillas=tuple(range(50)),   # más repeticiones para gráficas ricas
    tam_poblacion=50,            # un poco más de diversidad
    generaciones=80,             # más largo para estabilizar
    p_cruce=0.9,
    p_mutacion=0.25,
    elitismo=2,
    torneo_k=2,
    frac_cruce=0.3,
    p_inmigracion=0.02,
    k_candidatos_mut=5,
    usar_mutacion_informada=False,  # mantener rápido por defecto
    guardar_dir="resultados/genetico",
    medir_tiempo=True,
    outlier_umbral_s=5.0,
    verbose=False,
):
    """
    Ejecuta el GA varias veces, guarda tablas y graficos (estilo pruebasGRASP).
    """
    os.makedirs(guardar_dir, exist_ok=True)

    _, prep = cargar_y_preparar()

    runs = []
    Qs = []
    Ks = []
    tiempos = []
    mejor_global = {"Q": float("-inf"), "labels": None, "seed": None, "K": None}

    for rep, seed in enumerate(semillas):
        t0 = time.perf_counter()
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
            usar_mutacion_informada=usar_mutacion_informada,
        )
        t1 = time.perf_counter()

        K = len(np.unique(labels))
        dt = t1 - t0

        Qs.append(Q)
        Ks.append(K)
        tiempos.append(dt)

        best_last, avg_last = hist[-1]
        if verbose:
            print(
                f"GA rep={rep:>3} seed={seed:>3} | Q={Q:.6f} | K={K} | "
                f"last(best,avg)={best_last:.6f},{avg_last:.6f} | t={dt:.3f}s"
            )

        runs.append(
            {
                "algoritmo": "GA",
                "rep": rep,
                "seed": seed,
                "Q": float(Q),
                "K": int(K),
                "tiempo": float(dt),
                "tam_poblacion": tam_poblacion,
                "generaciones": generaciones,
                "p_cruce": p_cruce,
                "p_mutacion": p_mutacion,
                "elitismo": elitismo,
                "torneo_k": torneo_k,
                "frac_cruce": frac_cruce,
                "p_inmigracion": p_inmigracion,
                "k_candidatos_mut": k_candidatos_mut,
                "Q_last_best": float(best_last),
                "Q_last_avg": float(avg_last),
            }
        )

        if Q > mejor_global["Q"]:
            mejor_global.update(
                {"Q": float(Q), "labels": np.array(labels, copy=True), "seed": seed, "K": K}
            )

    Qs = np.array(Qs, dtype=float)
    Ks = np.array(Ks, dtype=int)
    tiempos = np.array(tiempos, dtype=float)

    print(
        f"GA | Q(mean+std)={Qs.mean():.6f}+{Qs.std():.6f} | Q(best)={Qs.max():.6f} | "
        f"K(mean)={Ks.mean():.2f} [{Ks.min()}..{Ks.max()}]"
    )
    if medir_tiempo and len(tiempos) > 0:
        print(
            f"GA | t(mean+std)={tiempos.mean():.3f}+{tiempos.std():.3f}s | "
            f"t(min..max)={tiempos.min():.3f}..{tiempos.max():.3f}s"
        )

    df_runs = pd.DataFrame(runs)
    if not df_runs.empty:
        df_summary = (
            df_runs["Q"]
            .agg(["mean", "std", "min", "max", "median"])
            .rename(
                {
                    "mean": "Q_mean",
                    "std": "Q_std",
                    "min": "Q_min",
                    "max": "Q_max",
                    "median": "Q_median",
                }
            )
            .to_frame()
            .T
        )

        path_runs = os.path.join(guardar_dir, "runs.csv")
        path_summary = os.path.join(guardar_dir, "summary.csv")
        df_runs.to_csv(path_runs, index=False)
        df_summary.to_csv(path_summary, index=False)

        with open(os.path.join(guardar_dir, "mejor.json"), "w", encoding="utf-8") as f:
            json.dump(
                {"Q": mejor_global["Q"], "seed": mejor_global["seed"], "K": mejor_global["K"]},
                f,
                ensure_ascii=False,
                indent=2,
            )

        plt.figure(figsize=(8, 6))
        sns.boxplot(data=df_runs, x="algoritmo", y="Q")
        plt.tight_layout()
        plt.savefig(os.path.join(guardar_dir, "boxplot_Q.png"))
        plt.close()

        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=df_runs, x="tiempo", y="Q", hue="seed", palette="viridis", s=25, edgecolor="none")
        plt.tight_layout()
        plt.savefig(os.path.join(guardar_dir, "scatter_Q_vs_tiempo.png"))
        plt.close()

        print(
            "\nGuardados:"
            f"\n - {path_runs}"
            f"\n - {path_summary}"
            "\n - mejor.json"
            "\n - boxplot_Q.png"
            "\n - scatter_Q_vs_tiempo.png"
        )


if __name__ == "__main__":
    probar_genetico(
        semillas=tuple(range(50)),
        tam_poblacion=50,
        generaciones=80,
        p_cruce=0.9,
        p_mutacion=0.25,
        elitismo=2,
        torneo_k=2,
        frac_cruce=0.3,
        p_inmigracion=0.02,
        k_candidatos_mut=5,
        usar_mutacion_informada=False,
    )
