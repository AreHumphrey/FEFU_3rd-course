import numpy as np


def power_iteration_method(A, epsilon=1e-7, max_iterations=10000):
    n = A.shape[0]
    x = np.random.rand(n)
    x = x / np.linalg.norm(x)
    lambda_old = 0

    for iteration in range(max_iterations):
        y = np.dot(A, x)
        x = y / np.linalg.norm(y)
        lambda_new = np.dot(x, np.dot(A, x))
        if abs(lambda_new - lambda_old) < epsilon:
            print(f"Метод степенных итераций завершён за {iteration + 1} итераций.")
            break
        lambda_old = lambda_new
    else:
        print("Метод степенных итераций не сошёлся за максимальное число итераций.")

    return lambda_new, x


def jacobi_rotation_method(A, epsilon=1e-7, max_iterations=10000):
    n = A.shape[0]
    A = A.copy()
    U = np.eye(n)

    def max_off_diagonal(matrix):
        max_value = 0
        max_idx = (0, 1)
        for i in range(n):
            for j in range(i + 1, n):
                if abs(matrix[i, j]) > abs(max_value):
                    max_value = matrix[i, j]
                    max_idx = (i, j)
        return max_value, max_idx

    for iteration in range(max_iterations):
        max_value, (i, j) = max_off_diagonal(A)
        if abs(max_value) < epsilon:
            print(f"Метод вращений Якоби завершён за {iteration + 1} итераций.")
            break

        phi = 0.5 * np.arctan2(2 * A[i, j], A[i, i] - A[j, j])
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)

        R = np.eye(n)
        R[i, i] = cos_phi
        R[j, j] = cos_phi
        R[i, j] = sin_phi
        R[j, i] = -sin_phi

        A = R.T @ A @ R
        U = U @ R
    else:
        print("Метод вращений Якоби не сошёлся за максимальное число итераций.")

    eigenvalues = np.diag(A)
    eigenvectors = U

    return eigenvalues, eigenvectors


if __name__ == "__main__":
    dimensions = [3, 5, 7]
    epsilons = [1e-3, 1e-7]

    for n in dimensions:
        print(f"Размер матрицы: {n}x{n}\n")
        A = np.random.rand(n, n)
        A = (A + A.T) / 2
        print("Сгенерированная матрица A:")
        print(A)

        for epsilon in epsilons:
            print(f"\nИспользуем точность: {epsilon}\n")

            eigenvalue, eigenvector = power_iteration_method(A, epsilon)
            print("Метод степенных итераций:")
            print(f"Собственное значение: {eigenvalue}")
            print(f"Собственный вектор: {eigenvector}\n")

            eigenvalues, eigenvectors = jacobi_rotation_method(A, epsilon)
            print("Метод вращений Якоби:")
            print(f"Собственные значения: {eigenvalues}")
            print(f"Собственные векторы: \n{eigenvectors}\n")

    print("\nПрограмма успешно завершена.")
