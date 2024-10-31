import numpy as np

A_main = np.array([
    [1, 2, 1, 4],
    [2, 0, 4, 3],
    [4, 2, 2, 1],
    [-3, 1, 3, 2]
], dtype=float)

b_main = np.array([13, 28, 20, 6], dtype=float)

A_I = np.array([
    [13.14, -2.12, 1.17, 0],
    [-2.12, 6.3, -2.45, 0],
    [1.17, -2.45, 4.6, 0],
    [0, 0, 0, 1]
], dtype=float)

b_I = np.array([1.27, 2.13, 3.14, 0], dtype=float)

A_II = np.array([
    [4.31, 0.26, 0.61, 0.27],
    [0.26, 2.32, 0.18, 0.34],
    [0.61, 0.18, 3.20, 0.31],
    [0.27, 0.34, 0.31, 5.17]
], dtype=float)

b_II = np.array([1.02, 1.00, 1.34, 1.27], dtype=float)


def lu_decomposition(A):
    n = A.shape[0]
    L = np.eye(n)
    U = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            U[i, j] = A[i, j] - sum(L[i, k] * U[k, j] for k in range(i))

        for j in range(i + 1, n):
            L[j, i] = (A[j, i] - sum(L[j, k] * U[k, i] for k in range(i))) / U[i, i]

    return L, U


def forward_substitution(L, b):
    n = L.shape[0]
    y = np.zeros(n)
    for i in range(n):
        y[i] = (b[i] - sum(L[i, j] * y[j] for j in range(i))) / L[i, i]
    return y


def backward_substitution(U, y):
    n = U.shape[0]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - sum(U[i, j] * x[j] for j in range(i + 1, n))) / U[i, i]
    return x


def solve_lu(A, b):
    L, U = lu_decomposition(A)
    y = forward_substitution(L, b)
    x = backward_substitution(U, y)
    return L, U, x


L_main, U_main, x_main = solve_lu(A_main, b_main)

L_I, U_I, x_I = solve_lu(A_I, b_I)

L_II, U_II, x_II = solve_lu(A_II, b_II)

print("Основная система:")
print("Матрица L:")
print(L_main)
print("\nМатрица U:")
print(U_main)
print("\nВектор решений x:")
print(x_main)

print("\nСистема I:")
print("Матрица L:")
print(L_I)
print("\nМатрица U:")
print(U_I)
print("\nВектор решений x:")
print(x_I)

print("\nСистема II:")
print("Матрица L:")
print(L_II)
print("\nМатрица U:")
print(U_II)
print("\nВектор решений x:")
print(x_II)
