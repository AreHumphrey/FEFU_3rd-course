import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import kagglehub


def load_and_prepare_data():
    path = kagglehub.dataset_download("gabrieltardochi/counter-strike-global-offensive-matches")

    file = next((f for f in os.listdir(path) if f.endswith('.csv')), None)
    file_path = os.path.join(path, file)
    df = pd.read_csv(file_path)

    print("\nинформация о наборе данных:")
    print(df.info())
    print("\nпервые строки:")
    print(df.head())

    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    df = pd.get_dummies(df)

    target = 'winner_t1'
    if target not in df.columns:
        raise ValueError(f"целевая переменная '{target}' не найдена в датасете")

    print(f"\nиспользуем в качестве целевой переменной: {target}")
    y = df[target].astype(int)
    df.drop(columns=[target], inplace=True)

    scaler = StandardScaler()
    X = scaler.fit_transform(df.values)

    return train_test_split(X, y, test_size=0.3, random_state=42)


def gaussian_rbf(x, c, s):
    return torch.exp(-torch.sum((x - c) ** 2, dim=1) / (2 * s ** 2))


def multiquadric_rbf(x, c, s):
    return torch.sqrt(torch.sum((x - c) ** 2, dim=1) + s ** 2)


class RBFLayer(nn.Module):
    def __init__(self, in_features, out_features, rbf_func='gaussian'):
        super().__init__()
        self.out_features = out_features
        self.centers = nn.Parameter(torch.randn(out_features, in_features), requires_grad=False)
        self.sigmas = nn.Parameter(torch.ones(out_features), requires_grad=False)
        self.rbf_func = gaussian_rbf if rbf_func == 'gaussian' else multiquadric_rbf

    def forward(self, x):
        return torch.stack([self.rbf_func(x, c, s) for c, s in zip(self.centers, self.sigmas)], dim=1)


class RBFNetwork(nn.Module):
    def __init__(self, in_features, rbf_units, out_features, rbf_func='gaussian'):
        super().__init__()
        self.rbf = RBFLayer(in_features, rbf_units, rbf_func)
        self.linear = nn.Linear(rbf_units, out_features)

    def forward(self, x):
        return self.linear(self.rbf(x))


def init_centers_kmeans(X, n_centers):
    kmeans = KMeans(n_clusters=n_centers).fit(X)
    return torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)


def compute_sigma_global(centers):
    from sklearn.metrics.pairwise import euclidean_distances
    distances = euclidean_distances(centers, centers)
    return np.mean(distances)


def train_model(model, loader, optimizer, criterion, epochs=50):
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            output = model(xb).squeeze()
            loss = criterion(output, yb)
            loss.backward()
            optimizer.step()


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_and_prepare_data()

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=32)

    centers = init_centers_kmeans(X_train, 20)
    sigma = compute_sigma_global(centers.numpy())

    model = RBFNetwork(X_train.shape[1], 20, 1)
    model.rbf.centers.data = centers
    model.rbf.sigmas.data = torch.full((20,), sigma)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCEWithLogitsLoss()

    train_model(model, train_loader, optimizer, criterion, epochs=30)

    model.eval()
    with torch.no_grad():
        preds = torch.sigmoid(model(X_test_tensor)).squeeze().numpy()
        preds_bin = (preds > 0.5).astype(int)

        print("\nоценка модели:")
        print(f"точность: {accuracy_score(y_test, preds_bin):.4f}")
        print(f"прецизионность: {precision_score(y_test, preds_bin):.4f}")
        print(f"полнота: {recall_score(y_test, preds_bin):.4f}")
        print(f"f1-мера: {f1_score(y_test, preds_bin):.4f}")

    print("\nисследование проклятия размерности:")
    dims = [5, 10, 20, 50, min(100, X_train.shape[1])]
    results = []

    for dim in dims:
        print(f"\nколичество признаков: {dim}")
        X_train_sub = X_train[:, :dim]
        X_test_sub = X_test[:, :dim]

        centers = init_centers_kmeans(X_train_sub, 20)
        sigma = compute_sigma_global(centers.numpy())

        model_sub = RBFNetwork(dim, 20, 1)
        model_sub.rbf.centers.data = centers
        model_sub.rbf.sigmas.data = torch.full((20,), sigma)

        loader = DataLoader(TensorDataset(torch.tensor(X_train_sub, dtype=torch.float32), y_train_tensor),
                            batch_size=32)
        optimizer = torch.optim.Adam(model_sub.parameters(), lr=0.01)

        train_model(model_sub, loader, optimizer, criterion, epochs=15)

        model_sub.eval()
        with torch.no_grad():
            preds = torch.sigmoid(model_sub(torch.tensor(X_test_sub, dtype=torch.float32))).squeeze().numpy()
            acc = accuracy_score(y_test, (preds > 0.5).astype(int))
            print(f"точность: {acc:.4f}")
            results.append(acc)

    plt.plot(dims, results, marker='o')
    plt.xlabel("количество признаков")
    plt.ylabel("точность")
    plt.title("влияние размерности на точность")
    plt.grid()
    plt.tight_layout()
    plt.show()

    print("\nсравнение функций rbf:")
    for func in ['gaussian', 'multiquadric']:
        print(f"\nфункция: {func}")
        net = RBFNetwork(X_train.shape[1], 20, 1, rbf_func=func)
        net.rbf.centers.data = init_centers_kmeans(X_train, 20)
        net.rbf.sigmas.data = torch.full((20,), sigma)

        train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=32)
        optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
        train_model(net, train_loader, optimizer, criterion, epochs=30)

        net.eval()
        with torch.no_grad():
            preds = torch.sigmoid(net(X_test_tensor)).squeeze().numpy()
            preds_bin = (preds > 0.5).astype(int)
            print(f"точность: {accuracy_score(y_test, preds_bin):.4f}")
            print(f"f1-мера: {f1_score(y_test, preds_bin):.4f}")

    print("\nсравнение с другими моделями:")
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier

    other_models = {
        "логистическая регрессия": LogisticRegression(max_iter=1000),
        "случайный лес": RandomForestClassifier(n_estimators=100)
    }

    for name, clf in other_models.items():
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        print(f"\n{name}")
        print(f"точность: {accuracy_score(y_test, preds):.4f}")
        print(f"f1-мера: {f1_score(y_test, preds):.4f}")
