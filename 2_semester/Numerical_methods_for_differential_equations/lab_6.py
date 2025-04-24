
import sympy as sp
import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

x = sp.symbols('x')

phi0 = 0
phi1 = (x + 1) * sp.ln(x + 1)
phi2 = x * (x - 1)
phi3 = x * (x - 1)**2

basis = [phi1, phi2, phi3]

basis_dbl_prime = [sp.diff(phi, x, 2) for phi in basis]
basis_prime = [sp.diff(phi, x) for phi in basis]

f_rhs = (x**2 + 2 * x + 2) / (x + 1)

def compute_integral(expr):
    func = sp.lambdify(x, expr, 'numpy')
    result, _ = quad(func, 0, 1)
    return result

n = len(basis)
A = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        L_phi_j = basis_dbl_prime[j] + (x + 1) * basis_prime[j] - basis[j]
        A[i, j] = compute_integral(L_phi_j * basis[i])

b = np.zeros(n)
for i in range(n):
    b[i] = compute_integral(f_rhs * basis[i])

C = np.linalg.solve(A, b)
u_approx = sum(C[i] * basis[i] for i in range(n))

u_exact_func = lambda x: (x + 1) * np.log(x + 1)
u_approx_func = sp.lambdify(x, u_approx, 'numpy')

x_vals = np.linspace(0, 1, 300)
u_exact_vals = u_exact_func(x_vals)
u_approx_vals = u_approx_func(x_vals)

plt.figure(figsize=(10, 6))
plt.plot(x_vals, u_approx_vals, label='Приближенное (Галёркин)', color='blue')
plt.plot(x_vals, u_exact_vals, label='Точное', color='red', linestyle='--')
plt.xlabel('x')
plt.ylabel('u(x)')
plt.title('Сравнение приближенного и точного решений (улучшенный базис)')
plt.legend()
plt.grid(True)
plt.show()

max_error = np.max(np.abs(u_exact_vals - u_approx_vals))
print(f'Максимальная ошибка: {max_error:.6f}')

u_approx_prime = sp.diff(u_approx, x)
u_approx_func_prime = sp.lambdify(x, u_approx_prime, 'numpy')

print('\nПроверка граничных условий:')
print(f'u(0) ≈ {u_approx_func(0):.6f} (должно быть 0)')
print(f'u(1) ≈ {u_approx_func(1):.6f} (должно быть ≈ {2 * np.log(2):.6f})')
