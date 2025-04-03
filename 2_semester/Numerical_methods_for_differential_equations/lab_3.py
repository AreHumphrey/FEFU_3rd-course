import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate

l = 1.0
T = 0.1  
M = 20  
N = 6  
a = 1.0 

h = l / M 
tau = T / (20 * N)  
sigma = a ** 2 * tau / h ** 2

x = np.linspace(0, l, M + 1)
t = np.linspace(0, T, N + 1)


def psi(x):
    return x * (1 - x)


def phi(x, t):
    return x * (x - x ** 2)


u_explicit = np.zeros((M + 1, N + 1))
u_implicit = np.zeros((M + 1, N + 1))
u_crank = np.zeros((M + 1, N + 1))

u_explicit[:, 0] = psi(x)
u_implicit[:, 0] = psi(x)
u_crank[:, 0] = psi(x)

u_explicit[0, :] = u_explicit[-1, :] = 0
u_implicit[0, :] = u_implicit[-1, :] = 0
u_crank[0, :] = u_crank[-1, :] = 0

for n in range(0, N):
    for i in range(1, M):
        u_explicit[i, n + 1] = u_explicit[i, n] + sigma * (
                    u_explicit[i + 1, n] - 2 * u_explicit[i, n] + u_explicit[i - 1, n]) + tau * phi(x[i], t[n])


A = np.diag((1 + 2 * sigma) * np.ones(M - 1)) + \
    np.diag(-sigma * np.ones(M - 2), 1) + \
    np.diag(-sigma * np.ones(M - 2), -1)

for n in range(0, N):
    b = u_implicit[1:M, n] + tau * phi(x[1:M], t[n])
    u_implicit[1:M, n + 1] = np.linalg.solve(A, b)

A_cn = np.diag((1 + sigma) * np.ones(M - 1)) + \
       np.diag(-sigma / 2 * np.ones(M - 2), 1) + \
       np.diag(-sigma / 2 * np.ones(M - 2), -1)

B_cn = np.diag((1 - sigma) * np.ones(M - 1)) + \
       np.diag(sigma / 2 * np.ones(M - 2), 1) + \
       np.diag(sigma / 2 * np.ones(M - 2), -1)

for n in range(0, N):
    b = B_cn @ u_crank[1:M, n] + tau * phi(x[1:M], t[n])
    u_crank[1:M, n + 1] = np.linalg.solve(A_cn, b)

plt.figure(figsize=(12, 8))
for n in range(N + 1):
    if n in [0, 2, 4, 6]:
        plt.plot(x, u_explicit[:, n], label=f'Явная схема (шаг {n})')
        plt.plot(x, u_implicit[:, n], label=f'Неявная схема (шаг {n})')
        plt.plot(x, u_crank[:, n], label=f'Кранк-Николсон (шаг {n})')

plt.title('Сравнение методов решения')
plt.xlabel('x')
plt.ylabel('u(x,t)')
plt.legend()
plt.grid(True)
plt.show()

results = {
    'x': x,
    'Явная схема (t=0)': u_explicit[:, 0],
    'Явная схема (t=6)': u_explicit[:, -1],
    'Неявная схема (t=0)': u_implicit[:, 0],
    'Неявная схема (t=6)': u_implicit[:, -1],
    'Кранк-Николсон (t=0)': u_crank[:, 0],
    'Кранк-Николсон (t=6)': u_crank[:, -1]
}

df = pd.DataFrame(results)
print(tabulate(df, headers='keys', tablefmt='grid'))
