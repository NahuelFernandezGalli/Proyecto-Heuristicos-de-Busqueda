# SQL
import sqlite3

# Pandas
import pandas as pd

# Graph
import community
import networkx as nx

# Plot
import matplotlib.pyplot as plt
import seaborn as sns

# Combinations
import itertools

# Numeric
import numpy as np


def crear_grafo():
    # Leer datos
    connect = sqlite3.connect("./Data_CDP/nips-papers/database.sqlite")
    query = """
    SELECT pa.paper_id, pa.author_id, a.name
    FROM paper_authors AS pa JOIN papers AS p ON pa.paper_id = p.id
    JOIN authors as a ON pa.author_id = a.id
    WHERE p.Year BETWEEN '2014' AND '2015'
    """
    df = pd.read_sql(query, connect)

    # Crear grafo
    G = nx.Graph()
    for p, a in df.groupby("paper_id")["name"]:
        for u, v in itertools.combinations(a, 2):
            if G.has_edge(u, v):
                G[u][v]["weight"] += 1
            else:
                G.add_edge(u, v, weight=1)

    return G


def preparar_grafo(G: nx.Graph, *, ordenar_nodos: bool = True):
    """
    Prepara estructuras auxiliares para heurísticas (GRASP / SA / GA).

    Devuelve un diccionario con:
      - nodes: lista fija de nodos (orden determinista si ordenar_nodos=True)
      - idx: dict nodo -> índice
      - n: número de nodos
      - m: suma de pesos total / 2  (m de modularidad ponderada)
      - strength: array (n,) con s_i = sum_j w_ij
      - neighbors: lista de listas con índices de vecinos (Si en el grafo existe la arista (u,v) = v es vecino de u)
      - weights: lista de dicts; weights[i][j] = w_ij (solo si hay arista)
    """
    if ordenar_nodos:
        try:
            nodes = sorted(G.nodes())
        except TypeError:
            # Por si hay nodos no comparables (poco probable aquí)
            nodes = sorted(G.nodes(), key=lambda x: str(x))
    else:
        nodes = list(G.nodes())

    idx = {u: i for i, u in enumerate(nodes)}
    n = len(nodes)

    strength = np.zeros(n, dtype=float)
    neighbors = [[] for _ in range(n)]
    weights = [dict() for _ in range(n)]

    total_w = 0.0
    for u, v, data in G.edges(data=True):
        w = float(data.get("weight", 1.0))
        iu, iv = idx[u], idx[v]

        # strength s_i
        strength[iu] += w
        strength[iv] += w

        # suma total de pesos (cada arista una vez)
        total_w += w

        # vecindario + pesos
        neighbors[iu].append(iv)
        neighbors[iv].append(iu)
        weights[iu][iv] = w
        weights[iv][iu] = w

    m = total_w / 2.0  # coherente con la definición (2m = suma de pesos)

    return {
        "nodes": nodes,
        "idx": idx,
        "n": n,
        "m": m,
        "strength": strength,
        "neighbors": neighbors,
        "weights": weights,
    }


def visualizar_grafo(G):
    plt.figure(figsize=(13, 9))
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, node_size=20, node_color="0.75")
    nx.draw_networkx_edges(G, pos, alpha=0.5, width=1)
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    # Crear el grafo
    G = crear_grafo()

    print("Número de autores en el grafo:", G.number_of_nodes())
    print("Número de colaboraciones (aristas) en el grafo:", G.number_of_edges())

    # Preparar estructuras para heurísticas
    prep = preparar_grafo(G)
    print("Preparación: ")
    print("n =", prep["n"])
    print("m =", prep["m"])
    print("Ejemplo nodos[0:5] =", prep["nodes"][:5])
    print("Ejemplo strength[0:5] =", prep["strength"][:5])

    # Visualizar el grafo
    visualizar_grafo(G)
