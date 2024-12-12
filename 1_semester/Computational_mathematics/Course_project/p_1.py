import numpy as np


def generate_matrix_A(n):
    return np.random.uniform(-1, 1, (n, n))


def generate_weights(n):
    return np.random.uniform(0.1, 1.0, n)


def construct_matrix_B(A, d):
    D_sqrt = np.sqrt(d)
    D_sqrt_inv = 1 / D_sqrt
    return A * np.outer(D_sqrt, D_sqrt_inv)


def calculate_subordinate_norm(A, d):
    B = construct_matrix_B(A, d)
    eigenvalues = np.linalg.eigvals(B)
    spectral_radius = max(np.abs(eigenvalues))
    return np.sqrt(spectral_radius)


n = 5

A = generate_matrix_A(n)
d = generate_weights(n)

M_A = calculate_subordinate_norm(A, d)

print(f"Матрица A:\n{A}")
print(f"Веса d:\n{d}")
print(f"Подчинённая матричная норма M(A): {M_A}")
