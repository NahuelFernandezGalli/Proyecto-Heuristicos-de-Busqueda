import numpy as np

from grafo import crear_grafo, preparar_grafo


def cargar_y_preparar():
    """Carga el grafo, lo prepara y muestra info basica."""
    print("Cargando grafo")
    G = crear_grafo()
    prep = preparar_grafo(G)

    print("Nodos:", prep["n"])
    print("Aristas:", G.number_of_edges())
    print()

    return G, prep


def resumen_Q_K(Qs, Ks, titulo="RESUMEN"):
    """Resumen comun de Q y numero de comunidades."""
    Qs = np.array(Qs, dtype=float)
    Ks = np.array(Ks, dtype=int)

    print(titulo)
    print(f"Q: mejor={Qs.max():.6f} | media={Qs.mean():.6f} | std={Qs.std():.6f} | peor={Qs.min():.6f}")
    print(f"K: mejor(Q)={Ks[Qs.argmax()]} | media={Ks.mean():.2f} | std={Ks.std():.2f} | min={Ks.min()} | max={Ks.max()}")
