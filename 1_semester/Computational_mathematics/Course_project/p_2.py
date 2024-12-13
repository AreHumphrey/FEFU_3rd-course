import numpy as np
import pandas as pd


def matrix_condition_number(A):
    return np.linalg.norm(A, ord=2) * np.linalg.norm(np.linalg.inv(A), ord=2)


def generate_matrices(size, perturbation):
    A = np.random.rand(size, size)
    while np.linalg.det(A) == 0:
        A = np.random.rand(size, size)

    B = A + perturbation * np.random.rand(size, size)
    while np.linalg.det(B) == 0:
        B = A + perturbation * np.random.rand(size, size)

    return A, B


def experimental_check(size, perturbation):
    A, B = generate_matrices(size, perturbation)

    A_inv = np.linalg.inv(A)
    B_inv = np.linalg.inv(B)

    left = np.linalg.norm(B_inv - A_inv, ord=2) / np.linalg.norm(B_inv, ord=2)
    mu_A = matrix_condition_number(A)
    right = mu_A * (np.linalg.norm(A - B, ord=2) / np.linalg.norm(A, ord=2))

    return A, B, left, right, left <= right


def multiple_experiments(size, perturbation, num_experiments):
    results = []
    for i in range(num_experiments):
        A, B, left, right, inequality_holds = experimental_check(size, perturbation)
        results.append({
            "Эксперимент": i + 1,
            "Матрица A": A,
            "Матрица B": B,
            "Левая часть (||B^{-1} - A^{-1}|| / ||B^{-1}||)": left,
            "Правая часть (μ(A) * ||A - B|| / ||A||)": right,
            "Неравенство выполняется?": inequality_holds
        })
    return results


size = 5
perturbation = 0.01
num_experiments = 3

results = multiple_experiments(size, perturbation, num_experiments)

formatted_results = []
for r in results:
    formatted_results.append({
        "Эксперимент": r["Эксперимент"],
        "Левая часть (||B^{-1} - A^{-1}|| / ||B^{-1}||)": r["Левая часть (||B^{-1} - A^{-1}|| / ||B^{-1}||)"],
        "Правая часть (μ(A) * ||A - B|| / ||A||)": r["Правая часть (μ(A) * ||A - B|| / ||A||)"],
        "Неравенство выполняется?": r["Неравенство выполняется?"],
        "Матрица A": r["Матрица A"],
        "Матрица B": r["Матрица B"]
    })

df_results = pd.DataFrame(formatted_results)

for _, row in df_results.iterrows():
    print(f"Эксперимент {row['Эксперимент']}:")
    print(f"Матрица A:\n{row['Матрица A']}")
    print(f"Матрица B:\n{row['Матрица B']}")
    print(f"Левая часть (||B^{-1} - A^{-1}|| / ||B^{-1}||): {row['Левая часть (||B^{-1} - A^{-1}|| / ||B^{-1}||)']}")
    print(f"Правая часть (μ(A) * ||A - B|| / ||A||): {row['Правая часть (μ(A) * ||A - B|| / ||A||)']}")
    print(f"Неравенство выполняется? {row['Неравенство выполняется?']}")
    print("-" * 50)
