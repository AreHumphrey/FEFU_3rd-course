import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

save_dir = "C:/Users/lutdi/PycharmProjects/chisel_diff/results"
os.makedirs(save_dir, exist_ok=True)

plot_path = os.path.join(save_dir, "comparison_plot.png")
csv_path = os.path.join(save_dir, "comparison_results.csv")


def f(x, y, a, c):
    return a / (c - x)


def modified_euler(f, x0, y0, h, x_end, a, c):
    x_values = np.arange(x0, x_end + h, h)
    y_values = np.zeros(len(x_values))
    y_values[0] = y0

    for n in range(len(x_values) - 1):
        k1 = f(x_values[n], y_values[n], a, c)
        k2 = f(x_values[n] + h / 2, y_values[n] + (h / 2) * k1, a, c)
        y_values[n + 1] = y_values[n] + h * k2

    return x_values, y_values


def analytical_solution(x, a, c):
    return a * np.log(c / (c - x))


x0, y0 = 0, 0
x_end = 1
h1 = 0.1
h2 = h1 / 2
a = 1
c = 2

x_h1, y_h1 = modified_euler(f, x0, y0, h1, x_end, a, c)
x_h2, y_h2 = modified_euler(f, x0, y0, h2, x_end, a, c)

y_exact = analytical_solution(x_h1, a, c)

plt.figure(figsize=(8, 5))
plt.plot(x_h1, y_h1, 'o-', label=f'Численный метод (h={h1})')
plt.plot(x_h2, y_h2, 's-', label=f'Численный метод (h={h2})')
plt.plot(x_h1, y_exact, 'k-', label='Аналитическое решение')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Сравнение численного и аналитического решений')
plt.legend()
plt.grid()
plt.savefig(plot_path)
plt.show()

comparison_data = {
    "x": x_h1,
    "y (численный, h)": y_h1,
    "y (численный, h/2)": y_h2[::2],
    "y (аналитическое)": y_exact
}

df = pd.DataFrame(comparison_data)

df = df.round(6)

df.to_csv(csv_path, index=False)

print(df)
