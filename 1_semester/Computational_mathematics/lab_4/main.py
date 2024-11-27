import numpy as np

A = np.array([[6.03, 13, -17],
              [13, 29.03, -38],
              [-17, -38, 50.03]])

b = np.array([2.0909, 4.1509, -5.1191])

x_exact = np.array([1.03, 1.03, 1.03])

Q, R = np.linalg.qr(A)

y = np.dot(Q.T, b)

x_computed = np.linalg.solve(R, y)

error = np.linalg.norm(x_computed - x_exact)

print("Результаты для исходной системы:")
print("Вычисленное решение x:", x_computed)
print("Точное решение x*:", x_exact)
print("Ошибка:", error)

A_new = np.array([[2, 0, 1],
                  [0, 2, 1],
                  [1, 1, 3]])
b_new = np.array([3, 0, 3])

Q_new, R_new = np.linalg.qr(A_new)

y_new = np.dot(Q_new.T, b_new)

x_new = np.linalg.solve(R_new, y_new)

print("\nРезультаты для новой системы:")
print("Вычисленное решение x:", x_new)

print("\nИтоги:")
print("Для исходной системы Ax = b:")
print("Решение x:", x_computed)
print("Ошибка относительно точного решения x*:", error)

print("\nДля новой системы:")
print("Решение x:", x_new)
