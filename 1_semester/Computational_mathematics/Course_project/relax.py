import numpy as np


def relaxation_method(A, b, omega=1.0, tol=1e-6, max_iter=3000, regularize=True):
    if A.shape[0] != A.shape[1]:
        raise ValueError("Матрица A должна быть квадратной.")
    if A.shape[0] != b.size:
        raise ValueError("Размерности матрицы A и вектора b должны совпадать.")
    if not (0 < omega <= 2):
        raise ValueError("Параметр релаксации omega должен быть в диапазоне (0, 2].")

    if regularize:
        diag_indices = np.diag_indices_from(A)
        A[diag_indices] += 1e-2 * np.max(np.abs(A))

    scale = np.linalg.norm(A, ord=np.inf)
    if scale > 0:
        A /= scale
        b /= scale

    n = A.shape[0]
    x = np.zeros(n, dtype=np.float64)
    converged = False

    for k in range(max_iter):
        x_new = np.copy(x)
        for i in range(n):
            if np.isclose(A[i, i], 0):
                raise ValueError(f"Нулевой диагональный элемент в строке {i}. Регуляризация может помочь.")

            residual = b[i] - np.dot(A[i, :], x_new) + A[i, i] * x_new[i]
            x_new[i] += omega * (residual / A[i, i])

            x_new[i] = np.clip(x_new[i], -1e3, 1e3)

            if np.isnan(x_new[i]) or np.isinf(x_new[i]):
                print(f"Ошибка: значение в строке {i} стало NaN или бесконечным на итерации {k}.")
                return x, k + 1, False

        diff = np.linalg.norm(x_new - x, ord=np.inf)
        norm_x_new = np.linalg.norm(x_new, ord=np.inf)
        if diff < tol or diff / max(norm_x_new, 1e-10) < tol:
            converged = True
            x = x_new
            break

        if k > 0 and k % 100 == 0:
            omega = max(0.9 * omega, 0.5)

        x = x_new

    if not converged:
        print(f"Метод не сошелся за {max_iter} итераций.")
    return x * scale, k + 1, converged


def solve_using_enlargement(A, b, omega=1.0, tol=1e-6, max_iter=3000, regularize=True):
    try:
        solution, iterations, converged = relaxation_method(A, b, omega, tol, max_iter, regularize)
        if not converged:
            print(f"Метод не сошелся за {iterations} итераций.")
        return solution
    except Exception as e:
        print(f"Ошибка: {e}")
        return None


def run_tests():
    print("# Тест 1: Плохо обусловленная матрица")
    A1 = np.array([[1, 2, 3],
                   [2, 4.01, 6],
                   [3, 6, 9.01]], dtype=float)
    b1 = np.array([6, 12.01, 18.01], dtype=float)
    solution1 = solve_using_enlargement(A1, b1)
    if solution1 is not None:
        expected1 = np.linalg.solve(A1, b1)
        absolute_error1 = np.abs(solution1 - expected1)
        relative_error1 = np.abs((solution1 - expected1) / expected1)
        print("Решение системы (плохо обусловленная матрица):", solution1)
        print("Абсолютная ошибка:", absolute_error1)
        print("Относительная ошибка:", relative_error1)
    print('')

    print("# Тест 2: Маленькая матрица (2x2)")
    A2 = np.array([[2, 1],
                   [1, 3]], dtype=float)
    b2 = np.array([3, 4], dtype=float)
    solution2 = solve_using_enlargement(A2, b2)
    if solution2 is not None:
        expected2 = np.linalg.solve(A2, b2)
        absolute_error2 = np.abs(solution2 - expected2)
        relative_error2 = np.abs((solution2 - expected2) / expected2)
        print("Решение системы (маленькая матрица):", solution2)
        print("Абсолютная ошибка:", absolute_error2)
        print("Относительная ошибка:", relative_error2)
    print('')

    print("# Тест 3: Единичная матрица")
    A3 = np.eye(3, dtype=float) * 1.05
    b3 = np.array([1.05, 2.1, 3.15], dtype=float)
    solution3 = solve_using_enlargement(A3, b3, omega=1.05, tol=1e-8, max_iter=3000)
    if solution3 is not None:
        expected3 = np.linalg.solve(A3, b3)
        absolute_error3 = np.abs(solution3 - expected3)
        relative_error3 = np.abs((solution3 - expected3) / expected3)
        print("Решение системы (единичная матрица):", solution3)
        print("Абсолютная ошибка:", absolute_error3)
        print("Относительная ошибка:", relative_error3)
    print('')

    print("# Тест 4: Симметричная положительно определённая матрица")
    A4 = np.array([[4, 1, 2],
                   [1, 3, 0],
                   [2, 0, 5]], dtype=float)
    b4 = np.array([7, 3, 8], dtype=float)
    solution4 = solve_using_enlargement(A4, b4)
    if solution4 is not None:
        expected4 = np.linalg.solve(A4, b4)
        absolute_error4 = np.abs(solution4 - expected4)
        relative_error4 = np.abs((solution4 - expected4) / expected4)
        print("Решение системы (симметричная положительно определённая матрица):", solution4)
        print("Абсолютная ошибка:", absolute_error4)
        print("Относительная ошибка:", relative_error4)
    print('')

    print("# Тест 5: Разреженная матрица (спарс-матрица)")
    A5 = np.array([[4, 0, 0],
                   [0, 3, 0],
                   [0, 0, 2]], dtype=float)
    b5 = np.array([4, 3, 2], dtype=float)
    solution5 = solve_using_enlargement(A5, b5, omega=1.0, tol=1e-8, max_iter=3000)
    if solution5 is not None:
        expected5 = np.linalg.solve(A5, b5)
        absolute_error5 = np.abs(solution5 - expected5)
        relative_error5 = np.abs((solution5 - expected5) / expected5)
        print("Решение системы (разреженная матрица):", solution5)
        print("Абсолютная ошибка:", absolute_error5)
        print("Относительная ошибка:", relative_error5)
    print('')

    print("# Тест 6: Вырожденная матрица")
    A6 = np.array([[1, 2, 3],
                   [2, 4, 6],
                   [3, 6, 9]], dtype=float)
    b6 = np.array([6, 12, 18], dtype=float)
    try:
        solution6 = solve_using_enlargement(A6, b6)
        if solution6 is not None:
            expected6 = np.linalg.solve(A6, b6)
            absolute_error6 = np.abs(solution6 - expected6)
            relative_error6 = np.abs((solution6 - expected6) / expected6)
            print("Решение системы (вырожденная матрица):", solution6)
            print("Абсолютная ошибка:", absolute_error6)
            print("Относительная ошибка:", relative_error6)
    except np.linalg.LinAlgError:
        print("Решение невозможно: матрица вырожденная.")
    print('')

    print("# Тест 7: Матрица с элементами разного порядка величин")
    A7 = np.array([[1e-3, 2], [2, 1e3]], dtype=float)
    b7 = np.array([2, 1002], dtype=float)
    solution7 = solve_using_enlargement(A7, b7)
    if solution7 is not None:
        expected7 = np.linalg.solve(A7, b7)
        absolute_error7 = np.abs(solution7 - expected7)
        relative_error7 = np.abs((solution7 - expected7) / expected7)
        print("Решение системы (разный порядок величин):", solution7)
        print("Абсолютная ошибка:", absolute_error7)
        print("Относительная ошибка:", relative_error7)
    print('')

    print("# Тест 8: Матрица с нулевыми элементами на диагонали")
    A8 = np.array([[1e-3, 1, 2],
                   [3, 1e-3, 4],
                   [5, 6, 1e-3]], dtype=float)
    b8 = np.array([3, 7, 11], dtype=float)
    solution8 = solve_using_enlargement(A8, b8, omega=1.0, tol=1e-8, max_iter=3000)
    if solution8 is not None:
        expected8 = np.linalg.solve(A8, b8)
        absolute_error8 = np.abs(solution8 - expected8)
        relative_error8 = np.abs((solution8 - expected8) / expected8)
        print("Решение системы (нулевые элементы на диагонали):", solution8)
        print("Абсолютная ошибка:", absolute_error8)
        print("Относительная ошибка:", relative_error8)
    print('')

    print("# Тест 9: Случайная матрица 5x5")
    np.random.seed(42)
    A9 = np.random.rand(5, 5) * 10
    A9 += np.eye(5) * 0.1
    b9 = np.random.rand(5) * 10
    print(A9, b9)
    solution9 = solve_using_enlargement(A9, b9, omega=1.05, tol=1e-8, max_iter=3000)
    if solution9 is not None:
        expected9 = np.linalg.solve(A9, b9)
        absolute_error9 = np.abs(solution9 - expected9)
        relative_error9 = np.abs((solution9 - expected9) / expected9)
        print("Решение системы (случайная матрица 5x5):", solution9)
        print("Абсолютная ошибка:", absolute_error9)
        print("Относительная ошибка:", relative_error9)
    print('')

    print("# Тест 10: Матрица с дробными элементами")
    A10 = np.array([[0.5, 0.25, 0.125], [0.25, 0.5, 0.125], [0.125, 0.125, 0.5]], dtype=float)
    b10 = np.array([1, 2, 3], dtype=float)
    solution10 = solve_using_enlargement(A10, b10)
    if solution10 is not None:
        expected10 = np.linalg.solve(A10, b10)
        absolute_error10 = np.abs(solution10 - expected10)
        relative_error10 = np.abs((solution10 - expected10) / expected10)
        print("Решение системы (дробные элементы):", solution10)
        print("Абсолютная ошибка:", absolute_error10)
        print("Относительная ошибка:", relative_error10)
    print('')

    print("# Тест 11: Матрица с отрицательными элементами")
    A11 = np.array([[4, -1, -2], [-1, 3, 0], [-2, 0, 5]], dtype=float)
    b11 = np.array([1, -2, 3], dtype=float)
    solution11 = solve_using_enlargement(A11, b11)
    if solution11 is not None:
        expected11 = np.linalg.solve(A11, b11)
        absolute_error11 = np.abs(solution11 - expected11)
        relative_error11 = np.abs((solution11 - expected11) / expected11)
        print("Решение системы (отрицательные элементы):", solution11)
        print("Абсолютная ошибка:", absolute_error11)
        print("Относительная ошибка:", relative_error11)
    print('')

    print("# Тест 12: Большая случайная матрица 10x10")
    np.random.seed(123)
    A12 = np.random.rand(10, 10) * 100
    b12 = np.random.rand(10) * 100
    print(A12)
    print(b12)
    solution12 = solve_using_enlargement(A12, b12)
    if solution12 is not None:
        expected12 = np.linalg.solve(A12, b12)
        absolute_error12 = np.abs(solution12 - expected12)
        relative_error12 = np.abs((solution12 - expected12) / expected12)
        print("Решение системы (большая случайная матрица 10x10):", solution12)
        print("Абсолютная ошибка:", absolute_error12)
        print("Относительная ошибка:", relative_error12)
    print('')

    print("# Тест 13: Идеально условная матрица")
    A13 = np.array([[2.1, 0], [0, 2.1]], dtype=float)
    b13 = np.array([4.2, 8.4], dtype=float)
    solution13 = solve_using_enlargement(A13, b13, omega=1.1, tol=1e-8, max_iter=3000)
    if solution13 is not None:
        expected13 = np.linalg.solve(A13, b13)
        absolute_error13 = np.abs(solution13 - expected13)
        relative_error13 = np.abs((solution13 - expected13) / expected13)
        print("Решение системы (идеально условная матрица):", solution13)
        print("Абсолютная ошибка:", absolute_error13)
        print("Относительная ошибка:", relative_error13)
    print('')

    print("# Тест 14: Диагональная матрица с дробными значениями")
    A14 = np.diag([0.5, 0.25, 0.125])
    b14 = np.array([1, 2, 3], dtype=float)
    solution14 = solve_using_enlargement(A14, b14)
    if solution14 is not None:
        expected14 = np.linalg.solve(A14, b14)
        absolute_error14 = np.abs(solution14 - expected14)
        relative_error14 = np.abs((solution14 - expected14) / expected14)
        print("Решение системы (диагональная матрица):", solution14)
        print("Абсолютная ошибка:", absolute_error14)
        print("Относительная ошибка:", relative_error14)
    print('')

    print("# Тест 15: Матрица с большим числом итераций для сходимости")
    A15 = np.array([[1, 0.5, 0.3], [0.5, 1, 0.1], [0.3, 0.1, 1]], dtype=float)
    b15 = np.array([1.8, 1.5, 1.2], dtype=float)
    solution15 = solve_using_enlargement(A15, b15)
    if solution15 is not None:
        expected15 = np.linalg.solve(A15, b15)
        absolute_error15 = np.abs(solution15 - expected15)
        relative_error15 = np.abs((solution15 - expected15) / expected15)
        print("Решение системы (медленная сходимость):", solution15)
        print("Абсолютная ошибка:", absolute_error15)
        print("Относительная ошибка:", relative_error15)
    print('')


def dop_test():
