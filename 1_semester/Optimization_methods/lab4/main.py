import numpy as np
from scipy.optimize import minimize


def generate_symmetric_matrix(n):
    matrix = np.random.rand(n, n)
    symmetric_matrix = (matrix + matrix.T) / 2
    return symmetric_matrix


def generate_random_vector(n):
    return np.random.rand(n) * 10


n = 4
A = generate_symmetric_matrix(n)
b = generate_random_vector(n)
x0 = np.array([2, 2, 2, 2])
r = 3


def f(x):
    return 0.5 * x.T @ A @ x + b @ x


def constraint(x):
    return r - np.linalg.norm(x - x0)


con = {'type': 'ineq', 'fun': constraint}

x_initial = np.array([0, 0, 0, 0])

result = minimize(f, x_initial, constraints=con)

if result.success:
    print("Минимум функции:", result.fun)
    print("Точка минимума:", result.x)
    print("Симметричная матрица A:\n", A)
    print("Вектор b:\n", b)
else:
    print("Оптимизация не завершена успешно.")
