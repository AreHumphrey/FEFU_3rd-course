import time
import pandas as pd
import numpy as np


def generate_spd_matrix(n):
    A = np.random.rand(n, n)
    A = np.dot(A, A.T)
    A += n * np.eye(n)
    return A


def relaxation_method(A, b, omega=1.0, tol=1e-6, max_iter=10000):
    n = A.shape[0]
    x = np.zeros(n)
    converged = False

    D = np.diag(np.diag(A))
    L = -np.tril(A, -1)
    U = -np.triu(A, 1)

    T = np.linalg.inv(D - omega * L) @ ((1 - omega) * D + omega * U)
    C = omega * np.linalg.inv(D - omega * L) @ b

    spectral_radius = max(abs(np.linalg.eigvals(T)))
    if spectral_radius >= 1:
        print("Предупреждение: Спектральный радиус >= 1. Метод может не сходиться.")

    for k in range(max_iter):
        x_new = T @ x + C
        if np.linalg.norm(x_new - x, np.inf) < tol:
            converged = True
            x = x_new
            break
        x = x_new

    if not converged:
        print(f"Метод не сошелся за {max_iter} итераций.")
    return x, k + 1, converged


def solve_using_sor(A, b, omega=1.0, tol=1e-6, max_iter=10000):
    try:
        solution, iterations, converged = relaxation_method(A, b, omega, tol, max_iter)
        if not converged:
            print(f"Метод не сошелся за {iterations} итераций.")
        return solution
    except Exception as e:
        print(f"Ошибка: {e}")
        return None


def measure_time(A, b, omega=1.0, tol=1e-6, max_iter=10000):
    start_time = time.perf_counter()
    solution = solve_using_sor(A, b, omega, tol, max_iter)
    end_time = time.perf_counter()
    return solution, end_time - start_time


sizes = [10, 100, 500, 1000, 5000, 10000]
time_results = []

for size in sizes:
    A = generate_spd_matrix(size)
    b = np.random.rand(size) * 10
    try:
        _, elapsed_time = measure_time(A, b, omega=1.25)
        time_results.append({
            "Размер матрицы": size,
            "Время выполнения (с)": elapsed_time
        })
    except ValueError as e:
        time_results.append({
            "Размер матрицы": size,
            "Время выполнения (с)": f"Ошибка: {str(e)}"
        })

time_results_df = pd.DataFrame(time_results)
print(time_results_df)
