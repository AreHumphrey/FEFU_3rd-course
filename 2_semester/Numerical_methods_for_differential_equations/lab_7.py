import numpy as np
from scipy.interpolate import BSpline
import matplotlib.pyplot as plt


def решить_граничную_задачу_методом_сплайнов():

    a, b = 0, 1

    k = 3

    n = 9

    num_basis = n + 2

    interior_knots = np.linspace(a, b, n)[1:-1]

    knots = np.concatenate([[a] * (k + 1), interior_knots, [b] * (k + 1)])

    collocation_points = []

    collocation_points.append(a)

    for i in range(len(interior_knots) - 1):
        collocation_points.append((interior_knots[i] + interior_knots[i + 1]) / 2)

    h = (b - a) / (n + 1)
    for i in range(n - len(interior_knots) + 1):
        collocation_points.append(a + (i + 1) * h)

    collocation_points.append(b)

    collocation_points = np.array(sorted(collocation_points[:num_basis]))

    A = np.zeros((num_basis, num_basis))
    F = np.zeros(num_basis)

    for j in range(1, num_basis - 1):
        x_val = collocation_points[j]

        for i in range(num_basis):
            coeff = np.zeros(num_basis)
            coeff[i] = 1
            spl = BSpline(knots, coeff, k)
            A[j, i] = spl.derivative(2)(x_val) + 0.5 * spl.derivative(1)(x_val) / (x_val + 1)

        F[j] = 1 / np.sqrt(x_val + 1)

    for i in range(num_basis):
        coeff = np.zeros(num_basis)
        coeff[i] = 1
        spl = BSpline(knots, coeff, k)
        A[0, i] = 3 * spl(a) - 2 * spl.derivative(1)(a)
    F[0] = 1

    for i in range(num_basis):
        coeff = np.zeros(num_basis)
        coeff[i] = 1
        spl = BSpline(knots, coeff, k)
        A[num_basis - 1, i] = spl.derivative(1)(b)
    F[num_basis - 1] = np.sqrt(2)

    cond = np.linalg.cond(A)
    print(f"Число обусловленности матрицы системы: {cond:.16f}")

    if cond > 1e15:
        print("Добавление регуляризации из-за высокого числа обусловленности")
        A = A + np.eye(A.shape[0]) * 1e-10

    try:
        c = np.linalg.solve(A, F)
    except np.linalg.LinAlgError:
        print("Обнаружена сингулярная матрица. Используется решение методом наименьших квадратов.")
        c, residuals, rank, s = np.linalg.lstsq(A, F, rcond=None)

    solution_spline = BSpline(knots, c, k)

    return solution_spline, knots, c, k


def точное_решение(x):

    return (2 / 3) * (x + 1) ** (3 / 2) + 1 / 3


def построить_результаты(solution_spline, knots, coeffs, k):

    x_plot = np.linspace(0, 1, 200)

    u_approx = solution_spline(x_plot)

    u_exact = точное_решение(x_plot)

    error = np.abs(u_approx - u_exact)
    max_error = np.max(error)

    rms_error = np.sqrt(np.mean(np.square(error)))

    plt.figure(figsize=(10, 6))
    plt.plot(x_plot, u_approx, 'r-', label='Приближенное')
    plt.plot(x_plot, u_exact, 'b--', label='Точное')
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.legend()
    plt.title(f'Сравнение решений (Среднеквадратичная погрешность: {rms_error:.16f})')
    plt.grid(True)
    plt.tight_layout()

    return max_error, rms_error


def main():

    solution, knots, coeffs, k = решить_граничную_задачу_методом_сплайнов()

    max_error, rms_error = построить_результаты(solution, knots, coeffs, k)

    print("\nКоэффициенты B-сплайна:")
    for i, c in enumerate(coeffs):
        print(f"c_{i} = {c:.16f}")

    print(f"\nМаксимальная погрешность: {max_error:.16f}")
    print(f"Среднеквадратичная погрешность: {rms_error:.16f}")

    x0, x1 = 0, 1
    u0 = solution(x0)
    u1 = solution(x1)
    u0_prime = solution.derivative(1)(x0)
    u1_prime = solution.derivative(1)(x1)

    print("\nПроверка граничных условий:")
    print(f"3u(0) - 2u'(0) = {3 * u0 - 2 * u0_prime:.16f} (должно быть 1)")
    print(f"u'(1) = {u1_prime:.16f} (должно быть {np.sqrt(2):.16f})")

    print("\nСравнение в конкретных точках:")
    check_points = [0, 0.25, 0.5, 0.75, 1]
    print("   x   |   Приблизительно   |   Точно       |   Погрешность")
    print("-------------------------------------------------------------")
    for x in check_points:
        approx = solution(x)
        exact = точное_решение(x)
        error = abs(approx - exact)
        print(f" {x:.2f}  | {approx:.16f} | {exact:.16f} | {error:.16f}")

    plt.show()


if __name__ == "__main__":
    main()
