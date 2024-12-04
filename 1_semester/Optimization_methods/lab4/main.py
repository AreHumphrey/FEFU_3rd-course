import numpy as np

np.random.seed(42)
A = np.random.rand(3, 3)
A = 0.5 * (A + A.T) + np.eye(3)
b = np.random.rand(3)
x0 = np.array([1.0, 1.0, 1.0])
r = 1.5

print("Сгенерированные данные:")
print("Матрица A:")
print(A)
print("\nВектор b:")
print(b)
print("\nЦентр сферы x0:")
print(x0)
print("\nРадиус сферы r:")
print(r)


def f(x):
    return 0.5 * x.T @ A @ x + b.T @ x


def grad_f(x):
    return A @ x + b


def project_to_ball(x, x0, r):
    diff = x - x0
    norm = np.linalg.norm(diff)
    if norm > r:
        return x0 + r * diff / norm
    return x


def gradient_descent(x0, r, alpha=0.1, tol=1e-6, max_iter=1000):
    x = np.random.rand(3)
    with open("gradient_descent_steps.txt", "w") as file:
        file.write("Шаги градиентного спуска:\n")
        for step in range(max_iter):
            grad = grad_f(x)
            x_new = x - alpha * grad
            x_new = project_to_ball(x_new, x0, r)
            file.write(f"Шаг {step + 1}: x = {x_new}, f(x) = {f(x_new)}\n")
            if np.linalg.norm(x_new - x) < tol:
                break
            x = x_new
    return x


def coordinate_descent(x0, r, tol=1e-6, max_iter=1000):
    x = np.random.rand(3)
    # print("\nПокоординатный спуск:")
    for step in range(max_iter):
        for i in range(len(x)):
            grad = grad_f(x)
            x[i] -= 0.1 * grad[i]
            x = project_to_ball(x, x0, r)
            # print(f"Шаг {step + 1}: x = {x}, f(x) = {f(x)}")
        if np.linalg.norm(grad_f(x)) < tol:
            break
    return x


def analytical_solution(A, b, x0, r):
    from scipy.optimize import minimize

    def lagrange(x):
        return f(x)

    constraint = {
        'type': 'eq',
        'fun': lambda x: np.linalg.norm(x - x0) - r
    }

    result = minimize(lagrange, np.random.rand(3), constraints=[constraint])
    return result.x


x_gd = gradient_descent(x0, r)
x_cd = coordinate_descent(x0, r)
x_an = analytical_solution(A, b, x0, r)

print("\nРезультаты:")
print(f"Градиентный спуск: x = {x_gd}, f(x) = {f(x_gd)}")
print(f"Покоординатный спуск: x = {x_cd}, f(x) = {f(x_cd)}")
print(f"Аналитическое решение: x = {x_an}, f(x) = {f(x_an)}")
