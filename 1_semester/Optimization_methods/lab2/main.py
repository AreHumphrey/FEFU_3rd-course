import numpy as np


def generate_positive_definite_matrix(size):
    M = np.random.rand(size, size)
    H = np.dot(M, M.T)
    return H


dimension = 3
H = generate_positive_definite_matrix(dimension)
c = np.random.rand(dimension)
x_start = np.zeros(dimension)

initial_step = 0.1
precision = 1e-6
max_steps = 1000
gradient_limit = 1e10


def gradient_descent(H, c, x_start, initial_step, precision, max_steps, gradient_limit):
    x = x_start
    step_size = initial_step
    iterations = 0
    path = [x]

    for _ in range(max_steps):
        grad = H @ x + c

        grad_norm = np.linalg.norm(grad)
        if grad_norm > gradient_limit:
            step_size *= 0.5
            print(f"Шаг уменьшен до {step_size} из-за большого градиента ({grad_norm})")

        x_new = x - step_size * grad
        path.append(x_new)

        if np.linalg.norm(x_new - x) < precision:
            break

        x = x_new
        iterations += 1

    return x, iterations, path


print("Сгенерированная матрица H:")
print(H)
print("\nСгенерированный вектор c:")
print(c)

x_min, steps, path = gradient_descent(H, c, x_start, initial_step, precision, max_steps, gradient_limit)

print("\nМинимум функции достигается в точке:", x_min)
print("Количество итераций:", steps)

for alpha in [0.01, 0.05, 0.1, 0.5, 1.0]:
    x_min, steps, _ = gradient_descent(H, c, x_start, alpha, precision, max_steps, gradient_limit)
    print(f"Шаг: {alpha}, Итерации: {steps}, Минимум в точке: {x_min}")
