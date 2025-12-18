import pandas as pd

from grafo import crear_grafo, preparar_grafo
from SAsolution import simulated_annealing_modularity


def correr_experimento_sa(
    reps=5,
    max_evals=100_000,
    ks=range(2, 101),
    T0=0.05,
    alpha=0.9995,
    seed0=123,
):
    print("Cargando grafo...")
    G = crear_grafo()
    prep = preparar_grafo(G)

    print("Nodos:", prep["n"])
    print("Aristas:", G.number_of_edges())
    print()

    rows = []
    for k in ks:
        for r in range(reps):
            seed = seed0 + 10_000 * k + r

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

            rows.append({"algo": "SA", "k": k, "rep": r, "seed": seed, "Q": Q_best})
            print(f"[SA] k={k:3d} rep={r:2d} seed={seed}  Q_best={Q_best:.6f}")

    df = pd.DataFrame(rows)
    summary = df.groupby("k")["Q"].agg(["mean", "std", "min", "max"]).reset_index()

    df.to_csv("results_sa_runs.csv", index=False)
    summary.to_csv("results_sa_summary.csv", index=False)

    print("\nGuardado:")
    print(" - results_sa_runs.csv")
    print(" - results_sa_summary.csv")


if __name__ == "__main__":
    correr_experimento_sa(
        reps=5,
        max_evals=100_000,
        T0=0.05,
        alpha=0.9995,
    )
