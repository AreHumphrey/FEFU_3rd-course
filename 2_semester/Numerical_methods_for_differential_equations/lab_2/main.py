

import numpy as np
import matplotlib.pyplot as plt


def solve_bvp(N, p, q, f, alpha0, beta0, gamma0, alpha1, beta1, gamma1):

    if N <= 1:
        raise ValueError("Число узлов N должно быть больше 1.")

    h = 1 / N 
    x = np.linspace(0, 1, N + 1) 

    a = np.zeros(N + 1)
    b = np.zeros(N + 1)
    c = np.zeros(N + 1)
    d = np.zeros(N + 1)

    for i in range(1, N):
        r = p(x[i]) * h / 2 
        correction = 1 + abs(r) ** 3 / (1 + abs(r) + r ** 2 - r)
        a[i] = correction / h ** 2 - p(x[i]) / (2 * h)
        b[i] = -2 * correction / h ** 2 + q(x[i])
        c[i] = correction / h ** 2 + p(x[i]) / (2 * h)
        d[i] = f(x[i])

    if beta0 == 0:  
        b[0] = alpha0
        d[0] = gamma0
    else: 
        b[0] = -3 / (2 * h) * beta0 + alpha0
        c[0] = 2 / (h) * beta0
        a[0] = -1 / (2 * h) * beta0
        d[0] = gamma0

    if beta1 == 0: 
        b[N] = alpha1
        d[N] = gamma1
    else:  
        b[N] = 3 / (2 * h) * beta1 + alpha1
        a[N] = -2 / (h) * beta1
        c[N] = 1 / (2 * h) * beta1
        d[N] = gamma1

    for i in range(1, N + 1):
        factor = a[i] / b[i - 1]
        b[i] -= factor * c[i - 1]
        d[i] -= factor * d[i - 1]

    y = np.zeros(N + 1)
    y[N] = d[N] / b[N]

    for i in range(N - 1, -1, -1):
        y[i] = (d[i] - c[i] * y[i + 1]) / b[i]

    return x, y, a, b, c, d

p = lambda x: x + 1  
q = lambda x: -1  
f = lambda x: (x ** 2 + 2 * x + 2) / (x + 1) 

N = 20  
alpha0, beta0, gamma0 = 1, 0, 0  
alpha1, beta1, gamma1 = 1, 0, 1.38294 

x, y, a, b, c, d = solve_bvp(N, p, q, f, alpha0, beta0, gamma0, alpha1, beta1, gamma1)

print("\nРешение краевой задачи методом Булеева-Тимухина")
print("-------------------------------------------------")
print(f"Число узлов: {N}")
print(f"Шаг сетки: {1 / N:.4f}")
print("\nЗначения решения в узлах:")
print("{:<10} {:<15} {:<15}".format("x", "u(x)", "Ошибка"))
print("-" * 40)
exact_solution = lambda x: (x + 1) * np.log(x + 1) 
error = np.abs(y - exact_solution(x))
for xi, yi, ei in zip(x, y, error):
    print(f"{xi:<10.4f} {yi:<15.6f} {ei:<15.6f}")

print("\nКоэффициенты разностной схемы:")
print("{:<10} {:<15} {:<15} {:<15} {:<15}".format("i", "a[i]", "b[i]", "c[i]", "d[i]"))
print("-" * 65)
for i in range(N + 1):
    print(f"{i:<10} {a[i]:<15.6f} {b[i]:<15.6f} {c[i]:<15.6f} {d[i]:<15.6f}")

plt.figure(figsize=(8, 5))
plt.plot(x, y, 'b-', label='Численное решение')
plt.plot(x, exact_solution(x), 'r--', label='Приблизительное точное решение')
plt.xlabel('x')
plt.ylabel('u(x)')
plt.title('Сравнение численного и точного решения')
plt.legend()
plt.grid()
plt.show()
