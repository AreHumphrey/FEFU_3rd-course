import numpy as np

np.random.seed(42)
A = np.random.randn(4, 4)
A = (A + A.T) / 2
b = np.random.randn(4)
x0 = np.random.randn(4)
r = np.random.uniform(1, 5)


def grad_f(x):
    return np.dot(A, x) + b


def hessian_f():
    return A


def is_within_constraint(x):
    return np.linalg.norm(x - x0) <= r


def project_to_sphere(x):
    direction = x - x0
    return x0 + direction / np.linalg.norm(direction) * r


def newton_method_with_constraints(f_grad, f_hessian, x_init, tol=1e-6, max_iter=100):
    x = x_init
    for i in range(max_iter):
        grad = f_grad(x)
        hessian = f_hessian()

        delta_x = np.linalg.solve(hessian, -grad)
        x_new = x + delta_x

        if not is_within_constraint(x_new):
            x_new = project_to_sphere(x_new)

        if np.linalg.norm(x_new - x) < tol:
            print(f"Сходимость достигнута за {i + 1} итераций.")
            break

        x = x_new

    return x


initial_guess = np.zeros(4)

optimal_x = newton_method_with_constraints(grad_f, hessian_f, initial_guess)
optimal_f = 0.5 * np.dot(optimal_x.T, np.dot(A, optimal_x)) + np.dot(b, optimal_x)

print("Матрица A:")
print("\n".join(["\t".join([f"{value:10.4f}" for value in row]) for row in A]))
print("\nВектор b:")
print("\t".join([f"{value:10.4f}" for value in b]))
print("\nЦентр ограничения x0:")
print("\t".join([f"{value:10.4f}" for value in x0]))
print(f"\nРадиус ограничения r: {r:.4f}")
print("\nОптимальное значение вектора x:")
print("\t".join([f"{value:10.4f}" for value in optimal_x]))
print(f"\nМинимальное значение функции f(x): {optimal_f:.4f}")
