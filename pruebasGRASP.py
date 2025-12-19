import os
import time
import json
import numpy as np
import networkx as nx
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from GRASPsolution import grasp_comunidades
from funcionObjetivo import modularidad_ponderada
from experimentos import cargar_y_preparar
from grafo import preparar_grafo


def _subgrafo_componente_gigante(G):
    comp = max(nx.connected_components(G), key=len)
    return G.subgraph(comp).copy()


def probar_grasp_por_k(
    k_min=2,
    k_max=50,
    repeticiones=20,
    alpha=0.3,
    rcl_k=None,
    permitir_merges_peores=True,
    usar_componente_gigante=True,
    verbose=False,
    medir_tiempo=True,
    trim_frac=0.10,
    outlier_umbral_s=5.0,
    guardar_dir="resultados/grasp",
):
    """
    Ejecuta GRASP para distintos K_obj, guarda tablas y graficos.
    """
    t_global0 = time.perf_counter()

    G, prep = cargar_y_preparar()

    if usar_componente_gigante:
        G = _subgrafo_componente_gigante(G)
        prep = preparar_grafo(G)

    n = prep["n"]
    num_componentes = nx.number_connected_components(G)

    print("Nodos:", n)
    print("Aristas:", G.number_of_edges())
    print("Componentes conexas:", num_componentes)
    print()

    os.makedirs(guardar_dir, exist_ok=True)

    resultados = {}
    runs = []
    mejor_global = {
        "Q": float("-inf"),
        "labels": None,
        "K_obj": None,
        "seed": None,
        "K_real": None,
    }

    for K_obj in range(k_min, k_max + 1):
        if K_obj < num_componentes:
            continue

        Qs = []
        Ks_reales = []
        tiempos = []
        outliers = 0

        max_merges = n - K_obj
        t_k0 = time.perf_counter()

        for rep in range(repeticiones):
            seed = rep

            t_rep0 = time.perf_counter()
            labels = grasp_comunidades(
                prep,
                alpha=alpha,
                rcl_k=rcl_k,
                max_merges=max_merges,
                semilla=seed,
                permitir_merges_peores=permitir_merges_peores,
            )
            t_rep1 = time.perf_counter()

            dt = t_rep1 - t_rep0
            if dt > outlier_umbral_s:
                outliers += 1

            Q = modularidad_ponderada(prep, labels)
            K_real = len(np.unique(labels))

            Qs.append(Q)
            Ks_reales.append(K_real)
            tiempos.append(dt)
            runs.append(
                {
                    "algoritmo": "GRASP",
                    "K_obj": K_obj,
                    "seed": seed,
                    "Q": float(Q),
                    "K_real": int(K_real),
                    "tiempo": float(dt),
                    "n": n,
                    "alpha": alpha,
                    "rcl_k": rcl_k,
                    "permitir_merges_peores": permitir_merges_peores,
                    "componente_gigante": usar_componente_gigante,
                }
            )

            if verbose:
                print(
                    f"K_obj={K_obj:>3} | rep={rep:>2} | seed={seed:>2} | "
                    f"Q={Q:.6f} | K_real={K_real} | t={dt:.3f}s"
                )

            if Q > mejor_global["Q"]:
                mejor_global.update(
                    {
                        "Q": float(Q),
                        "labels": np.array(labels, copy=True),
                        "K_obj": K_obj,
                        "seed": seed,
                        "K_real": K_real,
                    }
                )

        t_k1 = time.perf_counter()

        Qs = np.array(Qs, dtype=float)
        Ks_reales = np.array(Ks_reales, dtype=int)
        tiempos = np.array(tiempos, dtype=float)

        t_mediana = float(np.median(tiempos))

        tiempos_ord = np.sort(tiempos)
        cut = int(trim_frac * len(tiempos_ord))
        if len(tiempos_ord) > 2 * cut:
            tiempos_trim = tiempos_ord[cut:len(tiempos_ord) - cut]
        else:
            tiempos_trim = tiempos_ord
        t_mean_trim = float(tiempos_trim.mean())

        linea = (
            f"K_obj={K_obj:>3} | "
            f"Q(mean+std)={Qs.mean():.6f}+{Qs.std():.6f} | "
            f"Q(best)={Qs.max():.6f} | "
            f"K_real(mean)={Ks_reales.mean():.2f} "
            f"[{Ks_reales.min()}..{Ks_reales.max()}]"
        )

        if medir_tiempo:
            linea += (
                f" | t_rep(mean+std)={tiempos.mean():.3f}+{tiempos.std():.3f}s"
                f" | t_rep(mediana)={t_mediana:.3f}s"
                f" | t_rep(mean_trim)={t_mean_trim:.3f}s"
                f" | outliers(>{outlier_umbral_s:.1f}s)={outliers}/{repeticiones}"
                f" | t_K(total)={t_k1 - t_k0:.2f}s"
            )

        print(linea)

        resultados[K_obj] = {
            "Q": Qs,
            "K_real": Ks_reales,
            "Q_mean": float(Qs.mean()),
            "Q_std": float(Qs.std()),
            "Q_best": float(Qs.max()),
            "Q_worst": float(Qs.min()),
            "K_mean": float(Ks_reales.mean()),
            "K_min": int(Ks_reales.min()),
            "K_max": int(Ks_reales.max()),
        }

        if medir_tiempo:
            resultados[K_obj].update(
                {
                    "t_rep": tiempos,
                    "t_rep_mean": float(tiempos.mean()),
                    "t_rep_std": float(tiempos.std()),
                    "t_rep_min": float(tiempos.min()),
                    "t_rep_max": float(tiempos.max()),
                    "t_rep_mediana": t_mediana,
                    "t_rep_mean_trim": t_mean_trim,
                    "outliers": int(outliers),
                    "t_K_total": float(t_k1 - t_k0),
                }
            )

    if medir_tiempo:
        t_global1 = time.perf_counter()
        print(f"\nTiempo total experimento: {t_global1 - t_global0:.2f}s")

    if mejor_global["labels"] is not None:
        print(
            f"Mejor global: Q={mejor_global['Q']:.6f} | "
            f"K_obj={mejor_global['K_obj']} | "
            f"K_real={mejor_global['K_real']} | seed={mejor_global['seed']}"
        )

    df_runs = pd.DataFrame(runs)
    if not df_runs.empty:
        df_summary = (
            df_runs.groupby("K_obj")["Q"]
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

        path_runs = os.path.join(guardar_dir, "runs.csv")
        path_summary = os.path.join(guardar_dir, "summary.csv")
        df_runs.to_csv(path_runs, index=False)
        df_summary.to_csv(path_summary, index=False)

        with open(os.path.join(guardar_dir, "mejor.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "Q": mejor_global["Q"],
                    "K_obj": mejor_global["K_obj"],
                    "K_real": mejor_global["K_real"],
                    "seed": mejor_global["seed"],
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df_runs, x="K_obj", y="Q")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(guardar_dir, "boxplot_Q_por_K.png"))
        plt.close()

        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            data=df_runs,
            x="tiempo",
            y="Q",
            hue="K_obj",
            palette="viridis",
            s=25,
            edgecolor="none",
        )
        plt.tight_layout()
        plt.savefig(os.path.join(guardar_dir, "scatter_Q_vs_tiempo.png"))
        plt.close()

        print(
            "\nGuardados:"
            f"\n - {path_runs}"
            f"\n - {path_summary}"
            "\n - mejor.json"
            "\n - boxplot_Q_por_K.png"
            "\n - scatter_Q_vs_tiempo.png"
        )

    return resultados


if __name__ == "__main__":
    probar_grasp_por_k(
        k_min=2,
        k_max=50,
        repeticiones=20,
        alpha=0.30,
        rcl_k=None,
        permitir_merges_peores=True,
        usar_componente_gigante=True,
        verbose=False,
        medir_tiempo=True,
        trim_frac=0.10,
        outlier_umbral_s=5.0,
    )
