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

def crear_grafo():
    # Leer datos
    connect = sqlite3.connect('./Data_CDP/nips-papers/database.sqlite')
    query = """
    SELECT pa.paper_id, pa.author_id, a.name
    FROM paper_authors AS pa JOIN papers AS p ON pa.paper_id = p.id
    JOIN authors as a ON pa.author_id = a.id
    WHERE p.Year BETWEEN '2014' AND '2015'
    """
    df = pd.read_sql(query, connect)

    # Crear grafo
    G = nx.Graph()
    for p, a in df.groupby('paper_id')['name']: 
        for u, v in itertools.combinations(a, 2):
            if G.has_edge(u, v):
                G[u][v]['weight'] += 1
            else:
                G.add_edge(u, v, weight=1)

    return G



def visualizar_grafo(G):
    plt.figure(figsize=(13, 9))
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, node_size=20, node_color='0.75')
    nx.draw_networkx_edges(G, pos, alpha=0.5, width=1)
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    # Crear el grafo
    G = crear_grafo()

    # Info básica del grafo
    print("Número de autores en el grafo:", G.number_of_nodes())
    print("Número de colaboraciones (aristas) en el grafo:", G.number_of_edges())

    # Visualizar el grafo
    visualizar_grafo(G)