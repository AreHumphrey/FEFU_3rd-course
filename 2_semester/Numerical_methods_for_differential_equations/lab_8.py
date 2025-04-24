import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl

mpl.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12


def exact_solution(x, y, t, solution_type=1):

    if solution_type == 1:
        return t * np.exp(x + y)
    elif solution_type == 2:
        return t * np.sin(np.pi * x) * np.sin(np.pi * y)
    elif solution_type == 3:
        return t + x ** 2 + y ** 2
    elif solution_type == 4:
        return t + 0.25 * (x ** 2 + y ** 2)
    else:
        raise ValueError("Неверный тип решения")


def source_term(x, y, t, solution_type=1):

    if solution_type == 1:

        return np.exp(x + y) - t * np.exp(x + y) * 2
    elif solution_type == 2:

        return np.sin(np.pi * x) * np.sin(np.pi * y) + t * 2 * np.pi ** 2 * np.sin(np.pi * x) * np.sin(np.pi * y)
    elif solution_type == 3:

        return 1 - 4
    elif solution_type == 4:

        return 1 - 0.5
    else:
        raise ValueError("Неверный тип решения")


def fractional_step_method(solution_type=1, nx=10, ny=10, T=1.0, nt=100):
    lx, ly = 1.0, 1.0
    hx = lx / nx
    hy = ly / ny
    x = np.linspace(0, lx, nx + 1)
    y = np.linspace(0, ly, ny + 1)
    dt = T / nt

    Lambda1 = 1 / hx ** 2
    Lambda2 = 1 / hy ** 2

    # Начальное условие
    v = np.zeros((ny + 1, nx + 1))
    for i in range(ny + 1):
        for j in range(nx + 1):
            v[i, j] = exact_solution(x[j], y[i], 0, solution_type)

    for n in range(nt):
        t_n = n * dt
        t_np1 = (n + 1) * dt

        v_half = np.zeros_like(v)
        for i in range(1, ny):
            a = np.zeros(nx + 1)
            b = np.ones(nx + 1)
            c = np.zeros(nx + 1)
            d = np.zeros(nx + 1)

            for j in range(1, nx):
                phi = source_term(x[j], y[i], t_n, solution_type)
                a[j] = -dt * Lambda1 / 2
                b[j] = 1 + dt * Lambda1
                c[j] = -dt * Lambda1 / 2

                d[j] = v[i, j] + dt / 2 * (
                        Lambda1 * (v[i, j + 1] - 2 * v[i, j] + v[i, j - 1]) +
                        Lambda2 * (v[i + 1, j] - 2 * v[i, j] + v[i - 1, j]) + phi
                )

            # Границы по x
            mu_n = np.array([exact_solution(0, y[i], t_n, solution_type),
                             exact_solution(lx, y[i], t_n, solution_type)])
            mu_np1 = np.array([exact_solution(0, y[i], t_np1, solution_type),
                               exact_solution(lx, y[i], t_np1, solution_type)])
            mu_bar = mu_np1 - dt * Lambda2 * (mu_np1 - mu_n)

            b[0], d[0] = 1, mu_np1[0]
            b[nx], d[nx] = 1, mu_np1[1]

            v_half[i, :] = tridiagonal_solver(a, b, c, d, nx + 1)

        # Обновление граничных условий по y
        for j in range(nx + 1):
            v_half[0, j] = exact_solution(x[j], 0, t_np1, solution_type)
            v_half[ny, j] = exact_solution(x[j], ly, t_np1, solution_type)

        # Шаг 2: n+1/2 → n+1 по y
        v_new = np.zeros_like(v)
        for j in range(1, nx):
            a = np.zeros(ny + 1)
            b = np.ones(ny + 1)
            c = np.zeros(ny + 1)
            d = np.zeros(ny + 1)

            for i in range(1, ny):
                a[i] = -dt * Lambda2 / 2
                b[i] = 1 + dt * Lambda2
                c[i] = -dt * Lambda2 / 2

                d[i] = v_half[i, j] + dt * (
                        Lambda1 * (v_half[i, j + 1] - 2 * v_half[i, j] + v_half[i, j - 1])
                )

            b[0], d[0] = 1, exact_solution(x[j], 0, t_np1, solution_type)
            b[ny], d[ny] = 1, exact_solution(x[j], ly, t_np1, solution_type)

            v_new[:, j] = tridiagonal_solver(a, b, c, d, ny + 1)

        for i in range(ny + 1):
            v_new[i, 0] = exact_solution(0, y[i], t_np1, solution_type)
            v_new[i, nx] = exact_solution(lx, y[i], t_np1, solution_type)

        v = v_new.copy()

    u_exact = np.zeros_like(v)
    for i in range(ny + 1):
        for j in range(nx + 1):
            u_exact[i, j] = exact_solution(x[j], y[i], T, solution_type)

    return x, y, v, u_exact


def tridiagonal_solver(a, b, c, d, n):
    
    a_prime = np.copy(a)
    b_prime = np.copy(b)
    c_prime = np.copy(c)
    d_prime = np.copy(d)

    for i in range(1, n):
        m = a_prime[i] / b_prime[i - 1]
        b_prime[i] -= m * c_prime[i - 1]
        d_prime[i] -= m * d_prime[i - 1]

    x = np.zeros(n)
    x[n - 1] = d_prime[n - 1] / b_prime[n - 1]

    for i in range(n - 2, -1, -1):
        x[i] = (d_prime[i] - c_prime[i] * x[i + 1]) / b_prime[i]

    return x


def calculate_error(u_numerical, u_exact):

    diff = u_numerical - u_exact
    l2_error = np.sqrt(np.mean(diff ** 2))
    max_error = np.max(np.abs(diff))

    return l2_error, max_error


def plot_solution_2d(x, y, u_numerical, u_exact, solution_type):

    X, Y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(15, 10))  

    ax1 = fig.add_subplot(221, projection='3d') 
    surf1 = ax1.plot_surface(X, Y, u_numerical, cmap=cm.viridis, alpha=0.8)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('u')
    ax1.set_title('Численное решение')

    ax2 = fig.add_subplot(222, projection='3d')
    surf2 = ax2.plot_surface(X, Y, u_exact, cmap=cm.plasma, alpha=0.8)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('u')
    ax2.set_title('Точное решение')

    ax3 = fig.add_subplot(223, projection='3d')
    error = np.abs(u_numerical - u_exact)
    surf3 = ax3.plot_surface(X, Y, error, cmap=cm.hot, alpha=0.8)
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Ошибка')
    ax3.set_title('Абсолютная ошибка')

    ax4 = fig.add_subplot(224)
    mid_y = len(y) // 2
    ax4.plot(x, u_numerical[mid_y, :], 'b-', label='Численное')
    ax4.plot(x, u_exact[mid_y, :], 'r--', label='Точное')
    ax4.set_xlabel('X')
    ax4.set_ylabel('Значение u')
    ax4.set_title(f'Срез решения при y = {y[mid_y]:.2f}')
    ax4.legend()
    ax4.grid(True)

    plt.suptitle(f'Результаты метода расщепления (тип решения {solution_type})')
    plt.tight_layout(pad=3.0)  

    return fig


def solve_heat_equation():
    results = {}

    for solution_type in range(1, 5):
        print(f"\nРешение уравнения теплопроводности с типом решения {solution_type}...")
        nx, ny = 20, 20  
        T = 1.0  
        nt = 4000  

        x, y, u_numerical, u_exact = fractional_step_method(
            solution_type=solution_type,
            nx=nx,
            ny=ny,
            T=T,
            nt=nt
        )

        l2_error, max_error = calculate_error(u_numerical, u_exact)
        print(f"Ошибка L2: {l2_error}")
        print(f"Максимальная ошибка: {max_error}")

        fig = plot_solution_2d(x, y, u_numerical, u_exact, solution_type)
        image_filename = f"heat_solution_type_{solution_type}.png"
        fig.savefig(image_filename)
        print(f"График сохранён в файл: {image_filename}")

        txt_filename = f"heat_solution_type_{solution_type}_errors.txt"
        with open(txt_filename, 'w', encoding='utf-8') as f:
            f.write(f"Тип решения: {solution_type}\n")
            f.write(f"Ошибка L2: {l2_error}\n")
            f.write(f"Максимальная ошибка: {max_error}\n")
        print(f"Ошибки сохранены в файл: {txt_filename}")

        results[solution_type] = {
            'x': x,
            'y': y,
            'u_numerical': u_numerical,
            'u_exact': u_exact,
            'l2_error': l2_error,
            'max_error': max_error,
            'figure': fig
        }

    return results


if __name__ == "__main__":
    results = solve_heat_equation()

    for solution_type in range(1, 5):
        results[solution_type]['figure'].show()
    plt.show()
