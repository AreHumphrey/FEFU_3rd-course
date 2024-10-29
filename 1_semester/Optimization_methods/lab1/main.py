from math import sin


def f(x):
    return x ** 2 / 2 - sin(x)


n = 0
k = 0


def bisection_method(a_, b_, epsilon):
    global n
    a = a_
    b = b_

    while abs(b - a) > epsilon:
        n += 1
        c = (a + b) / 2
        fa = f(a)
        fc = f(c)

        if fa * fc < 0:
            b = c
        else:
            a = c

    return f((a + b) / 2)


phi = 1.6180339887


def golden_mean(a_, b_, epsilon):
    global k
    a = a_
    b = b_

    x1 = b - (b - a) / phi
    x2 = a + (b - a) / phi
    f1 = f(x1)
    f2 = f(x2)

    while abs(b - a) > epsilon:
        k += 1
        if f1 > f2:
            a = x2
            x2 = x1
            f2 = f1
            x1 = a + (b - a) / phi
            f1 = f(x1)
        else:
            b = x1
            x1 = x2
            f1 = f2
            x2 = b - (b - a) / phi
            f2 = f(x2)

    return f((a + b) / 2)


print(bisection_method(0, 1, 0.03), golden_mean(0, 1, 0.03))

import matplotlib.pyplot as plt
from math import log

epsilons = [0.1, 0.05, 0.03, 0.01, 0.005, 0.001, 0.0005, 0.0001]

dichotomy_calls = []
golden_calls = []

for eps in epsilons:
    n = 0
    k = 0

    bisection_method(0, 1, eps)
    dichotomy_calls.append(n)

    golden_mean(0, 1, eps)
    golden_calls.append(k)

plt.figure(figsize=(10, 6))
plt.step([log(eps) for eps in epsilons], dichotomy_calls, label="Dichotomy Method", where='mid', color='blue')
plt.step([log(eps) for eps in epsilons], golden_calls, label="Golden Section Method", where='mid', color='red')

plt.xlabel(r'$\ln(\epsilon)$', fontsize=18, fontstyle='italic')
plt.ylabel(r'Количество вызовов $n$', fontsize=18, fontstyle='italic')
plt.title("Сравнение количества вызовов для методов", fontsize=20, fontweight='bold')
plt.legend()
plt.grid(True)

plt.xticks(fontsize=12, fontstyle='italic')
plt.yticks(fontsize=12, fontstyle='italic')

plt.show()

