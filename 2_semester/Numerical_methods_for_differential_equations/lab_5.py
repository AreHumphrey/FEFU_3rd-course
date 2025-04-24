import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

print("Решение дифференциального уравнения методом коллокации:")
print("u'' + (x + 1) * u' - u = (x^2 + 2x + 2)/(x + 1)")
print("с граничными условиями: u(0) = 0, u(1) = 2ln(2)")

x = sp.Symbol('x')
c0, c1, c2, c3 = sp.symbols('c0 c1 c2 c3')

u = c0 + c1 * x + c2 * x ** 2 + c3 * x ** 3

u_prime = sp.diff(u, x)
u_double_prime = sp.diff(u_prime, x)

f = (x**2 + 2 * x + 2) / (x + 1)

R = u_double_prime + (x + 1) * u_prime - u - f

eq1 = u.subs(x, 0) - 0               
eq2 = u.subs(x, 1) - 2 * np.log(2)  

collocation_points = [0.3, 0.7]
eq3 = R.subs(x, collocation_points[0])
eq4 = R.subs(x, collocation_points[1])

equations = [eq1, eq2, eq3, eq4]
variables = [c0, c1, c2, c3]

A = np.zeros((4, 4))
b = np.zeros(4)

for i, eq in enumerate(equations):
    for j, var in enumerate(variables):
        coeff = eq.coeff(var) if var in eq.free_symbols else 0
        A[i, j] = float(coeff)
    const = eq.subs({v: 0 for v in variables})
    b[i] = -float(const)

coeffs = np.linalg.solve(A, b)
c0_val, c1_val, c2_val, c3_val = coeffs

print("\nКоэффициенты приближенного решения:")
print(f"c0 = {c0_val:.6f}")
print(f"c1 = {c1_val:.6f}")
print(f"c2 = {c2_val:.6f}")
print(f"c3 = {c3_val:.6f}")

def u_approx(x_val):
    return c0_val + c1_val * x_val + c2_val * x_val ** 2 + c3_val * x_val ** 3

def u_exact(x_val):
    return (x_val + 1) * np.log(x_val + 1)

print("\nТочное решение: u(x) = (x + 1) * ln(x + 1)")

print("\nСравнение точного и приближенного решений:")
points = [0, 0.25, 0.5, 0.75, 1]
for p in points:
    approx = u_approx(p)
    exact = u_exact(p)
    print(f"x = {p:.2f}: приближ. = {approx:.6f}, точн. = {exact:.6f}, ошибка = {approx - exact:.6f}")

x_vals = np.linspace(0, 1, 200)
u_approx_vals = u_approx(x_vals)
u_exact_vals = u_exact(x_vals)

plt.figure(figsize=(10, 6))
plt.plot(x_vals, u_approx_vals, label='Приближенное решение', color='blue')
plt.plot(x_vals, u_exact_vals, label='Точное решение', color='red', linestyle='--')
plt.xlabel('x')
plt.ylabel('u(x)')
plt.title('Сравнение приближенного и точного решений')
plt.legend()
plt.grid(True)
plt.show()

error = np.abs(u_exact_vals - u_approx_vals)
print(f"\nМаксимальная ошибка: {np.max(error):.6f}")
