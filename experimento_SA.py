import os
import json
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from grafo import crear_grafo, preparar_grafo
from SAsolution import simulated_annealing_modularity


def correr_experimento_sa(
    reps=10,
    max_evals=100_000,
    ks=range(2, 101),
    T0=0.05,
    alpha=0.9995,
    seed0=123,
    guardar_dir="resultados/sa",
):
    """
    Ejecuta SA varias veces y guarda tablas y graficos.
    """
    os.makedirs(guardar_dir, exist_ok=True)

    print("Cargando grafo...")
    G = crear_grafo()
    prep = preparar_grafo(G)

    print("Nodos:", prep["n"])
    print("Aristas:", G.number_of_edges())
    print()

    rows = []
    mejor = {"Q": float("-inf"), "k": None, "seed": None}

    for k in ks:
        for r in range(reps):
            seed = seed0 + 10_000 * k + r

            t0 = time.perf_counter()
            labels_best, Q_best = simulated_annealing_modularity(
                prep,
                k=k,
                max_evals=max_evals,
                T0=T0,
                alpha=alpha,
                seed=seed,
                enforce_nonempty=True,
                return_history=False,
            )
            t1 = time.perf_counter()

            rows.append(
                {
                    "algoritmo": "SA",
                    "k": k,
                    "rep": r,
                    "seed": seed,
                    "Q": Q_best,
                    "tiempo": t1 - t0,
                    "n": prep["n"],
                    "max_evals": max_evals,
                    "T0": T0,
                    "alpha": alpha,
                }
            )
            print(f"[SA] k={k:3d} rep={r:2d} seed={seed}  Q_best={Q_best:.6f} t={t1 - t0:.2f}s")

            if Q_best > mejor["Q"]:
                mejor.update({"Q": float(Q_best), "k": k, "seed": seed})

    df = pd.DataFrame(rows)
    summary = (
        df.groupby("k")["Q"]
        .agg(["mean", "std", "min", "max", "median"])
        .reset_index()
        .rename(
            columns={
                "mean": "Q_mean",
                "std": "Q_std",
                "min": "Q_min",
                "max": "Q_max",
                "median": "Q_median",
            }
        )
    )

    print(
        f"SA | Q(mean+std)={df['Q'].mean():.6f}+{df['Q'].std():.6f} | "
        f"Q(best)={df['Q'].max():.6f}"
    )
    print(f"Mejor global: Q={mejor['Q']:.6f} | k={mejor['k']} | seed={mejor['seed']}")

    path_runs = os.path.join(guardar_dir, "results_sa_runs.csv")
    path_summary = os.path.join(guardar_dir, "results_sa_summary.csv")
    df.to_csv(path_runs, index=False)
    summary.to_csv(path_summary, index=False)
    with open(os.path.join(guardar_dir, "mejor.json"), "w", encoding="utf-8") as f:
        json.dump(mejor, f, ensure_ascii=False, indent=2)

    # Graficos
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="k", y="Q")
    plt.xticks(rotation=90)
    plt.xlabel("k objetivo")
    plt.ylabel("Modularidad ponderada (Q)")
    plt.tight_layout()
    plt.savefig(os.path.join(guardar_dir, "boxplot_Q_por_k.png"))
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x="tiempo", y="Q", hue="k", palette="viridis", s=25, edgecolor="none")
    plt.xlabel("Tiempo de ejecución (s)")
    plt.ylabel("Modularidad ponderada (Q)")
    plt.tight_layout()
    plt.savefig(os.path.join(guardar_dir, "scatter_Q_vs_tiempo.png"))
    plt.close()

    print("\nGuardado:")
    print(f" - {path_runs}")
    print(f" - {path_summary}")
    print(" - mejor.json")
    print(" - boxplot_Q_por_k.png")
    print(" - scatter_Q_vs_tiempo.png")


if __name__ == "__main__":
    correr_experimento_sa(
        reps=20,
        max_evals=100_000,
        T0=0.05,
        alpha=0.9995,
    )
