import numpy as np


def sqrt_method(A, b):
    n = A.shape[0]
    U = np.zeros_like(A, dtype=float)

    for i in range(n):
        U[i, i] = np.sqrt(A[i, i] - np.sum(U[:i, i] ** 2))
        for j in range(i + 1, n):
            U[i, j] = (A[i, j] - np.sum(U[:i, i] * U[:i, j])) / U[i, i]

    y = np.zeros(n)
    for i in range(n):
        y[i] = (b[i] - np.sum(U[:i, i] * y[:i])) / U[i, i]

    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.sum(U[i, i + 1:] * x[i + 1:])) / U[i, i]

    return x


A = np.array([
    [1, 2, 4],
    [2, 13, 23],
    [4, 23, 77]
], dtype=float)

b = np.array([10, 50, 150], dtype=float)

x = sqrt_method(A, b)
print("Решение x*:", x)

A1 = np.array([
    [5.8, 0.3, -0.2],
    [0.3, 4.0, -0.7],
    [-0.2, -0.7, 6.7]
], dtype=float)

b1 = np.array([3.1, -1.7, 1.1], dtype=float)

x1 = sqrt_method(A1, b1)
print("Решение для СЛАУ I x*:", x1)

A2 = np.array([
    [4.12, 0.42, 1.34, 0.88],
    [0.42, 3.95, 1.87, 0.43],
    [1.34, 1.87, 3.20, 0.31],
    [0.88, 0.43, 0.31, 5.17]
], dtype=float)

b2 = np.array([11.17, 0.115, 9.909, 9.349], dtype=float)

x2 = sqrt_method(A2, b2)
print("Решение для СЛАУ II x*:", x2)
