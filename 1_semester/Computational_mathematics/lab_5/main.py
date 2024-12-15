import numpy as np


def jacobi_method(A, b, tol=1e-4, max_iterations=1000):
    n = len(b)
    x = np.zeros(n)
    x_new = np.zeros(n)
    iterations = 0

    while True:
        for i in range(n):
            x_new[i] = (b[i] - sum(A[i, j] * x[j] for j in range(n) if j != i)) / A[i, i]
        iterations += 1

        if np.linalg.norm(x_new - x, ord=np.inf) < tol or iterations >= max_iterations:
            break
        x = x_new.copy()

    return x_new, iterations


def sor_method(A, b, omega, tol=1e-4, max_iterations=1000):
    n = len(b)
    x = np.zeros(n)
    iterations = 0

    while True:
        x_old = x.copy()
        for i in range(n):
            sigma = sum(A[i, j] * x[j] for j in range(i)) + sum(A[i, j] * x_old[j] for j in range(i + 1, n))
            x[i] = (1 - omega) * x_old[i] + (omega / A[i, i]) * (b[i] - sigma)
        iterations += 1

        if np.linalg.norm(x - x_old, ord=np.inf) < tol or iterations >= max_iterations:
            break

    return x, iterations


def main():
    A = np.array([[6.22, 1.42, -1.72, 1.91],
                  [1.42, 5.33, 1.11, -1.82],
                  [-1.72, 1.11, 5.24, 1.42],
                  [1.91, -1.82, 1.42, 6.55]])
    b = np.array([7.53, 6.06, 8.05, 8.06])
    tol = 1e-4

    print("Решение методом Якоби:")
    jacobi_solution, jacobi_iterations = jacobi_method(A, b, tol)
    print(f"Решение: {jacobi_solution}")
    print(f"Количество итераций: {jacobi_iterations}")

    omega = 1.1
    print("\nРешение методом верхней релаксации (SOR):")
    sor_solution, sor_iterations = sor_method(A, b, omega, tol)
    print(f"Решение: {sor_solution}")
    print(f"Количество итераций: {sor_iterations}")

    print("\nСравнение сходимости:")
    print(f"Метод Якоби: {jacobi_iterations} итераций")
    print(f"Метод верхней релаксации: {sor_iterations} итераций")


if __name__ == "__main__":
    main()
