import math
import random
import time
import os
import tsplib95


def load_solutions(solutions_path: str) -> dict:
    """Читает файл solutions (формат 'name : value') -> {name: optimum}."""
    if not os.path.exists(solutions_path):
        raise FileNotFoundError(f"Файл solutions не найден: '{solutions_path}'")

    solutions = {}
    with open(solutions_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            name, _, raw_value = line.partition(":")
            value_str = raw_value.split("(")[0].strip()
            try:
                solutions[name.strip()] = int(value_str)
            except ValueError:
                pass

    return solutions


def load_tsp_file(tsp_dir: str, name: str):
    """Загружает .tsp файл из папки tsp_dir по имени инстанса."""
    path = os.path.join(tsp_dir, f"{name}.tsp")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Файл '{path}' не найден.")
    return tsplib95.load(path)


def build_distance_matrix(problem) -> tuple:
    """Строит матрицу расстояний dist[i][j] (индексы 0..n-1). Возвращает (dist, n)."""
    nodes = list(problem.get_nodes())
    n = len(nodes)
    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                dist[i][j] = problem.get_weight(nodes[i], nodes[j])
    return dist, n


def tour_length(tour: list, dist: list) -> float:
    """Длина замкнутого тура: F(S) = sum dist[S[i]][S[i+1]] + dist[S[n-1]][S[0]]."""
    n = len(tour)
    return sum(dist[tour[i]][tour[(i + 1) % n]] for i in range(n))


def get_neighbor(tour: list) -> list:
    """
    Возвращает случайного соседа S' из N(S) — оператор 2-opt:
    выбирает два случайных индекса i < j и переворачивает сегмент [i..j].
    """
    n = len(tour)
    i, j = sorted(random.sample(range(n), 2))
    return tour[:i] + tour[i:j + 1][::-1] + tour[j + 1:]


def simulated_annealing(
    dist: list,
    n: int,
    T_start: float      = 1000.0,
    r: float            = 0.995,
    L: int              = 150,
    T_min: float        = 1e-3,
    max_no_improve: int = 300,
    verbose: bool       = True,
) -> tuple:
    """
    Шаг 3.1: L раз выбирается один случайный сосед S' из N(S),
    решение S обновляется сразу при каждом принятии.
    Параметры: T_start, r, L, T_min, max_no_improve.
    Возвращает (best_tour, best_length).
    """

    # Шаг 1: начальное решение S, вычислить F(S)
    S   = list(range(n))
    random.shuffle(S)
    F_S = tour_length(S, dist)

    best_tour   = S[:]
    best_length = F_S

    # Шаг 2: задать начальную температуру T
    T = T_start
    no_improve_count = 0
    iteration = 0

    if verbose:
        print(f"\n  Параметры: T_start={T_start}, r={r}, L={L}, "
              f"T_min={T_min}, max_no_improve={max_no_improve}")
        print(f"  Начальная длина тура: {F_S:.1f}")
        print(f"\n  {'Итер':>7}  {'T':>10}  {'F(S)':>10}  {'Лучшее':>10}")
        print(f"  {'-'*44}")

    # Шаг 3: основной цикл
    while T > T_min and no_improve_count < max_no_improve:
        improved_this_epoch = False

        # Шаг 3.1: выполнить цикл L раз
        for _ in range(L):

            # i.  Выбрать в N(S) случайным образом решение S'
            S_prime   = get_neighbor(S)
            F_S_prime = tour_length(S_prime, dist)

            # ii. Δ = F(S') - F(S)
            delta = F_S_prime - F_S

            if delta <= 0:
                # iii. Δ ≤ 0 — принять безусловно
                S   = S_prime
                F_S = F_S_prime
                if F_S < best_length:
                    best_tour   = S[:]
                    best_length = F_S
                    improved_this_epoch = True
            else:
                # iv. Δ > 0 — принять с вероятностью exp(-Δ/T)
                if random.random() < math.exp(-delta / T):
                    S   = S_prime
                    F_S = F_S_prime

        # Шаг 3.2: понизить температуру T := T * r
        T  *= r
        iteration += 1

        if improved_this_epoch:
            no_improve_count = 0
        else:
            no_improve_count += 1

        if verbose and iteration % 200 == 0:
            print(f"  {iteration:>7}  {T:>10.4f}  {F_S:>10.1f}  {best_length:>10.1f}")

    if verbose:
        print(f"\n  Остановка: итерации={iteration}, T={T:.6f}, "
              f"no_improve={no_improve_count}")

    return best_tour, best_length


if __name__ == "__main__":
    random.seed(42)

    TSP_DIR        = "tsp"
    SOLUTIONS_FILE = "solutions.txt"
    INSTANCE_NAMES = [
        "pa561",
    ]

    solutions = load_solutions(SOLUTIONS_FILE)
    results   = {}

    for name in INSTANCE_NAMES:
        problem = load_tsp_file(TSP_DIR, name)
        optimum = solutions.get(name)

        print(f"\n{'='*55}")
        print(f"  Инстанс : {name}   (городов: {problem.dimension})")
        # if optimum is not None:
        #     print(f"  Оптимум : {optimum}")
        print(f"{'='*55}")

        dist, n = build_distance_matrix(problem)

        t0 = time.time()
        best_tour, best_length = simulated_annealing(dist, n, verbose=True)
        elapsed = time.time() - t0

        print(f"\n  >>> Лучшая длина тура : {best_length:.1f}")
        # if optimum is not None:
        #     gap = (best_length - optimum) / optimum * 100
        #     print(f"  >>> Отклонение от опт.: {gap:.2f}%")
        print(f"  >>> Время работы      : {elapsed:.2f} сек")
        print(f"  >>> Маршрут (города)  : {[c + 1 for c in best_tour]}")

        results[name] = best_length

    print(f"\n\n{'='*60}")
    print("  ИТОГОВАЯ ТАБЛИЦА")
    print(f"{'='*60}")
    print(f"  {'Тестовый набор':<16} {'Алгоритм':<28} {'ЦФ':>10}")
    print(f"  {'-'*57}")
    for inst_name, val in results.items():
        print(f"  {inst_name:<16} {'Имитация отжига (SA)':<28} {val:>10.1f}")
    print(f"{'='*60}")

# T_start=1000.0, r=0.995, L=500, T_min=1e-3, max_no_improve=300
# T_start=1000.0, r=0.999, L=500, T_min=1e-3, max_no_improve=500
