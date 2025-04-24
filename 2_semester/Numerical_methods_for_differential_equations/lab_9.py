import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

def f(x):
    return 1 - x

def alpha1(x): return x
def alpha2(x): return x ** 2

def beta1(s): return (1 + s) * s
def beta2(s): return (1 + s) * s ** 2

def calculate_f_i(i):
    if i == 1:
        return quad(lambda s: beta1(s) * f(s), 0, 1)[0]
    elif i == 2:
        return quad(lambda s: beta2(s) * f(s), 0, 1)[0]

def calculate_A_ij(i, j):
    alpha = [alpha1, alpha2]
    beta = [beta1, beta2]
    coeffs = [0.2, 0.02]

    return quad(lambda s: beta[i - 1](s) * alpha[j - 1](s), 0, 1)[0] * coeffs[i - 1]

f1 = calculate_f_i(1)
f2 = calculate_f_i(2)

A11 = calculate_A_ij(1, 1)
A12 = calculate_A_ij(1, 2)
A21 = calculate_A_ij(2, 1)
A22 = calculate_A_ij(2, 2)

A = np.array([
    [1 - A11, -A12],
    [-A21, 1 - A22]
])
b = np.array([f1, f2])
C1, C2 = np.linalg.solve(A, b)

print(f"Коэффициенты системы:")
print(f"f1 = {f1:.8f}")
print(f"f2 = {f2:.8f}")
print(f"A11 = {A11:.8f}")
print(f"A12 = {A12:.8f}")
print(f"A21 = {A21:.8f}")
print(f"A22 = {A22:.8f}")
print(f"C1 = {C1:.8f}")
print(f"C2 = {C2:.8f}")

def u_approx(x):
    return f(x) + C1 * alpha1(x) + C2 * alpha2(x)

u_exact = u_approx

x_values = np.linspace(0, 1, 101)
u_approx_values = [u_approx(x) for x in x_values]
u_exact_values = [u_exact(x) for x in x_values]
errors = [abs(a - b) for a, b in zip(u_approx_values, u_exact_values)]
max_error = max(errors)

print(f"\nМаксимальная погрешность (по сравнению с точным): {max_error:.8f}")

def residual(x):
    integrand = lambda s: (1 + s) * (np.exp(0.2 * x * s) - 1) * u_approx(s)
    integral, _ = quad(integrand, 0, 1)
    return u_approx(x) - (f(x) + integral)

residuals = [residual(x) for x in x_values]
max_residual = max(abs(r) for r in residuals)

print(f"Максимальный остаток: {max_residual:.8f}")

plt.figure(figsize=(10, 8))

plt.subplot(3, 1, 1)
plt.plot(x_values, u_approx_values, 'b-', label='Приближенное решение')
plt.plot(x_values, u_exact_values, 'r--', label='Точное решение')
plt.title('Приближенное и точное решения')
plt.xlabel('x')
plt.ylabel('u(x)')
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(x_values, residuals, 'g-', label='Остаток')
plt.title('Остаток приближенного решения')
plt.xlabel('x')
plt.ylabel('Остаток')
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(x_values, errors, 'purple', label='Абсолютная погрешность')
plt.title('Абсолютная погрешность (приближенное vs точное)')
plt.xlabel('x')
plt.ylabel('Погрешность')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

print("\nЗначения в контрольных точках:")
for x in [0, 0.5, 1]:
    ua = u_approx(x)
    ue = u_exact(x)
    print(f"x = {x}: u_approx = {ua:.8f}, u_exact = {ue:.8f}, error = {abs(ua - ue):.8f}")

