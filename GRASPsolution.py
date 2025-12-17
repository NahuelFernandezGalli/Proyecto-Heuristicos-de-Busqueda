from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
import random

import numpy as np


# ============================================================
# 1) Representación de la solución
# ============================================================

@dataclass
class Solucion:
    # asignacion[i] = id de comunidad del nodo i, o -1 si no asignado
    asignacion: np.ndarray  # shape (n,), dtype=int
    # número de comunidades actuales (ids válidos: 0 .. k-1)
    k: int


def crear_solucion_vacia(n: int) -> Solucion:
    asignacion = np.full(n, -1, dtype=int)
    return Solucion(asignacion=asignacion, k=0)


def solucion_completa(sol: Solucion) -> bool:
    return np.all(sol.asignacion >= 0)


# ============================================================
# 2) Función objetivo (placeholder)
# ============================================================
# Objetivo temporal:
#   maximizar el peso intra-comunidad total
# Se implementa como minimización:
#   f = - peso_intra

def funcion_objetivo_placeholder(sol: Solucion, prep: Dict) -> float:
    pesos = prep["weights"]
    n = prep["n"]
    a = sol.asignacion

    peso_intra = 0.0
    for i in range(n):
        ci = a[i]
        for j, w in pesos[i].items():
            if j > i and a[j] == ci:
                peso_intra += w

    return -peso_intra


def peso_intra_total(sol: Solucion, prep: Dict) -> float:
    return -funcion_objetivo_placeholder(sol, prep)


# ============================================================
# 3) Construcción GRASP (greedy-randomized)
# ============================================================

@dataclass(frozen=True)
class Movimiento:
    nodo: int
    comunidad_destino: int


@dataclass
class ConfiguracionGRASP:
    max_iteraciones: int = 200
    tam_rcl: int = 10
    semilla: int = 0

    # construcción
    iniciar_en_nodo_fuerte: bool = True
    permitir_nueva_comunidad: bool = True

    # búsqueda local
    usar_busqueda_local: bool = True
    max_pasos_busqueda_local: int = 5000


def elegir_nodo_semilla(prep: Dict, asignado: np.ndarray,
                        rng: random.Random,
                        *, nodo_fuerte: bool) -> int:
    if nodo_fuerte:
        fuerza = prep["strength"]
        max_f = np.max(fuerza)
        candidatos = np.where(fuerza == max_f)[0]
        return int(rng.choice(list(candidatos)))

    libres = np.where(~asignado)[0]
    return int(rng.choice(list(libres)))


def nodos_frontera(prep: Dict, asignado: np.ndarray) -> np.ndarray:
    vecinos = prep["neighbors"]
    n = prep["n"]

    frontera = np.zeros(n, dtype=bool)
    for u in np.where(asignado)[0]:
        for v in vecinos[u]:
            if not asignado[v]:
                frontera[v] = True

    nodos = np.where(frontera)[0]
    if nodos.size == 0:
        nodos = np.where(~asignado)[0]

    return nodos


def comunidades_candidatas(nodo: int, sol: Solucion, prep: Dict) -> List[int]:
    vecinos = prep["neighbors"]
    a = sol.asignacion

    comms = set()
    for j in vecinos[nodo]:
        cj = a[j]
        if cj >= 0:
            comms.add(int(cj))

    return sorted(comms)


def puntuacion_construccion(nodo: int, comunidad: int,
                            sol: Solucion, prep: Dict) -> float:
    vecinos = prep["neighbors"]
    pesos = prep["weights"]
    a = sol.asignacion

    score = 0.0
    for j in vecinos[nodo]:
        if a[j] == comunidad:
            score += pesos[nodo].get(j, 0.0)

    return score


def aplicar_movimiento(sol: Solucion, mov: Movimiento) -> None:
    sol.asignacion[mov.nodo] = mov.comunidad_destino


def construir_solucion_greedy_aleatoria(
    sol: Solucion,
    prep: Dict,
    cfg: ConfiguracionGRASP,
    rng: random.Random,
) -> Solucion:
    n = prep["n"]
    asignado = np.zeros(n, dtype=bool)

    # ---- nodo semilla ----
    semilla = elegir_nodo_semilla(
        prep, asignado, rng,
        nodo_fuerte=cfg.iniciar_en_nodo_fuerte
    )
    sol.k = 1
    sol.asignacion[semilla] = 0
    asignado[semilla] = True

    # ---- construcción incremental ----
    while not np.all(asignado):
        frontera = nodos_frontera(prep, asignado)
        i = int(rng.choice(list(frontera)))

        comms = comunidades_candidatas(i, sol, prep)
        candidatos: List[Tuple[float, Movimiento]] = []

        if not comms:
            if cfg.permitir_nueva_comunidad:
                candidatos.append((0.0, Movimiento(i, sol.k)))
            if sol.k > 0:
                c = rng.randrange(sol.k)
                candidatos.append((0.0, Movimiento(i, c)))
        else:
            for c in comms:
                s = puntuacion_construccion(i, c, sol, prep)
                candidatos.append((s, Movimiento(i, c)))

            if cfg.permitir_nueva_comunidad:
                candidatos.append((0.0, Movimiento(i, sol.k)))

        candidatos.sort(key=lambda x: x[0], reverse=True)
        k = max(1, min(cfg.tam_rcl, len(candidatos)))
        rcl = candidatos[:k]

        _, elegido = rng.choice(rcl)

        if elegido.comunidad_destino == sol.k:
            sol.k += 1

        aplicar_movimiento(sol, elegido)
        asignado[i] = True

    return sol


# ============================================================
# 4) Búsqueda local (simple)
# ============================================================

def busqueda_local_mover_nodo(
    sol: Solucion,
    prep: Dict,
    funcion_objetivo: Callable[[Solucion, Dict], float],
    cfg: ConfiguracionGRASP,
) -> Solucion:
    rng = random.Random(cfg.semilla + 99991)
    n = prep["n"]
    vecinos = prep["neighbors"]

    mejor = Solucion(sol.asignacion.copy(), sol.k)
    mejor_valor = funcion_objetivo(mejor, prep)

    pasos = 0
    while pasos < cfg.max_pasos_busqueda_local:
        mejora = False
        orden = list(range(n))
        rng.shuffle(orden)

        for i in orden:
            ci = int(mejor.asignacion[i])

            comms = set(int(mejor.asignacion[j]) for j in vecinos[i])
            comms.discard(ci)

            for c2 in comms:
                candidata = Solucion(mejor.asignacion.copy(), mejor.k)
                candidata.asignacion[i] = c2
                val = funcion_objetivo(candidata, prep)

                if val < mejor_valor:
                    mejor, mejor_valor = candidata, val
                    mejora = True
                    break

            if mejora:
                break

        if not mejora:
            break

        pasos += 1

    return mejor


# ============================================================
# 5) GRASP principal
# ============================================================

def grasp(
    prep: Dict,
    cfg: ConfiguracionGRASP,
    funcion_objetivo: Optional[Callable[[Solucion, Dict], float]] = None,
) -> Tuple[Solucion, float]:

    if funcion_objetivo is None:
        funcion_objetivo = funcion_objetivo_placeholder

    rng = random.Random(cfg.semilla)
    n = prep["n"]

    mejor_sol: Optional[Solucion] = None
    mejor_val: Optional[float] = None

    for _ in range(cfg.max_iteraciones):
        sol = crear_solucion_vacia(n)
        sol = construir_solucion_greedy_aleatoria(sol, prep, cfg, rng)

        if cfg.usar_busqueda_local:
            sol = busqueda_local_mover_nodo(sol, prep, funcion_objetivo, cfg)

        val = funcion_objetivo(sol, prep)

        if mejor_val is None or val < mejor_val:
            mejor_sol, mejor_val = sol, val

    return mejor_sol, mejor_val


# ============================================================
# 6) Ejemplo de uso
# ============================================================

if __name__ == "__main__":
    # Ajusta el import al nombre real de tu archivo de grafo
    #
    # from grafo_cdp import crear_grafo, preparar_grafo
    #
    # G = crear_grafo()
    # prep = preparar_grafo(G)
    #
    # cfg = ConfiguracionGRASP(max_iteraciones=50, tam_rcl=10, semilla=0)
    # sol, val = grasp(prep, cfg)
    #
    # print("Comunidades:", sol.k)
    # print("Objetivo (placeholder):", val)
    # print("Peso intra:", peso_intra_total(sol, prep))
    #
    raise SystemExit("Configura el import de crear_grafo / preparar_grafo y ejecuta.")
