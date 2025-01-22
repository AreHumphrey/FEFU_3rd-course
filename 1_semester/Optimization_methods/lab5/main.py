import numpy as np

def simplex_method(c, A, b):

    c = -c  # Преобразование задачи максимизации в задачу минимизации.
    num_vars = A.shape[1]  # Количество переменных.
    num_constraints = A.shape[0]  # Количество ограничений.

    tableau = np.zeros((num_constraints + 1, num_vars + num_constraints + 1))

    tableau[:-1, :num_vars] = A  # Матрица ограничений A.
    tableau[:-1, num_vars:num_vars + num_constraints] = np.eye(num_constraints)  # Единичная матрица для базисных переменных.
    tableau[:-1, -1] = b  # Вектор b в правой части.
    tableau[-1, :num_vars] = c  # Коэффициенты целевой функции.

    # Итерации симплекс-метода:
    while np.any(tableau[-1, :-1] < 0):  # Пока есть отрицательные элементы в строке целевой функции.
        pivot_col = np.argmin(tableau[-1, :-1])  # Находим ведущий столбец.

        # Рассчитываем отношения для выбора ведущей строки:
        pivot_elements = tableau[:-1, pivot_col]
        ratios = np.full(len(pivot_elements), np.inf)  # Начальные значения отношений.
        valid_rows = pivot_elements > 0  # Отбираем только положительные элементы столбца.
        ratios[valid_rows] = tableau[:-1, -1][valid_rows] / pivot_elements[valid_rows]  # Отношения b[i] / A[i][pivot_col].

        if np.all(ratios == np.inf):  # Проверяем, есть ли ограничение задачи.
            raise ValueError("Задача не ограничена: решение не существует.")

        pivot_row = np.argmin(ratios)  # Ведущая строка определяется минимальным отношением.

        # Обновляем таблицу:
        tableau[pivot_row, :] /= tableau[pivot_row, pivot_col]  # Нормируем ведущую строку.
        for i in range(tableau.shape[0]):  # Обновляем остальные строки.
            if i != pivot_row:
                tableau[i, :] -= tableau[i, pivot_col] * tableau[pivot_row, :]

    # Вычисляем оптимальное решение:
    solution = np.zeros(num_vars)
    for i in range(num_vars):
        col = tableau[:-1, i]
        if np.count_nonzero(col) == 1 and np.sum(col) == 1:  # Проверяем базисные переменные.
            row = np.where(col == 1)[0][0]
            solution[i] = tableau[row, -1]  # Присваиваем значение из правой части.

    optimal_value = -tableau[-1, -1]  # Оптимальное значение целевой функции.
    return optimal_value, solution


c = np.array([3, 2, 4])
A = np.array([
    [2, 1, 1],
    [1, 2, 3],
    [2, 2, 1]
])
b = np.array([8, 10, 8])

try:
    optimal_value, solution = simplex_method(c, A, b)

    print("Оптимальное значение целевой функции:", optimal_value)
    print("Оптимальное решение:", solution)
except ValueError as e:
    print("Ошибка:", e)
