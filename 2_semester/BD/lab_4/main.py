import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("train.csv")
df["rain"] = (df["rainfall"] > 0).astype(int)

X = df[["pressure", "maxtemp", "temparature", "mintemp", "dewpoint",
        "humidity", "cloud", "sunshine"]]
y = df["rain"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

class_counts = y_train.value_counts().to_dict()
total = sum(class_counts.values())
weights = [total / class_counts[i] for i in range(2)]
class_weights = torch.tensor(weights, dtype=torch.float32)


class MLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.model(x)


mlp = MLP(X_train.shape[1])
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(mlp.parameters(), lr=0.001)

losses = []
for epoch in range(100):
    optimizer.zero_grad()
    output = mlp(X_train_tensor)
    loss = criterion(output, y_train_tensor)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    if epoch % 10 == 0:
        print(f"Эпоха {epoch}: Потери = {loss.item():.4f}")

plt.plot(losses)
plt.title("MLP Потери при обучении")
plt.xlabel("Эпоха")
plt.ylabel("Потери")
plt.grid()
plt.show()

mlp.eval()
with torch.no_grad():
    y_pred = mlp(X_test_tensor)
    _, predicted = torch.max(y_pred, 1)
    report_dict = classification_report(y_test_tensor, predicted, output_dict=True, zero_division=0)
    labels_map = {"0": "Без дождя", "1": "Дождь"}
    print("\n=== Отчёт по классификации MLP ===")
    for label in ["0", "1"]:
        print(f"{labels_map[label]}:")
        print(f"  Точность:  {report_dict[label]['precision']:.2f}")
        print(f"  Полнота:   {report_dict[label]['recall']:.2f}")
        print(f"  F-мера:    {report_dict[label]['f1-score']:.2f}")
    print(f"Общая точность: {report_dict['accuracy']:.2f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test_tensor, predicted)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Без дождя", "Дождь"],
                yticklabels=["Без дождя", "Дождь"])
    plt.xlabel("Предсказание")
    plt.ylabel("Истинное значение")
    plt.title("Матрица ошибок MLP")
    plt.show()

SEQ_LEN = 7
sequences, targets = [], []
for i in range(len(X_scaled) - SEQ_LEN):
    window = X_scaled[i:i + SEQ_LEN]
    rain_days = y.iloc[i:i + SEQ_LEN].sum()
    label = int(rain_days >= 3)
    sequences.append(window)
    targets.append(label)

X_seq = np.array(sequences)
y_seq = np.array(targets)

X_seq_train, X_seq_test, y_seq_train, y_seq_test = train_test_split(
    X_seq, y_seq, test_size=0.2, random_state=42, stratify=y_seq
)

X_seq_train_tensor = torch.tensor(X_seq_train, dtype=torch.float32)
X_seq_test_tensor = torch.tensor(X_seq_test, dtype=torch.float32)
y_seq_train_tensor = torch.tensor(y_seq_train, dtype=torch.long)
y_seq_test_tensor = torch.tensor(y_seq_test, dtype=torch.long)


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=32):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)


lstm = LSTMModel(input_size=X_seq.shape[2])
criterion_lstm = nn.CrossEntropyLoss()
optimizer_lstm = optim.Adam(lstm.parameters(), lr=0.001)

lstm_losses = []
for epoch in range(30):
    optimizer_lstm.zero_grad()
    output = lstm(X_seq_train_tensor)
    loss = criterion_lstm(output, y_seq_train_tensor)
    loss.backward()
    optimizer_lstm.step()
    lstm_losses.append(loss.item())
    if epoch % 5 == 0:
        print(f"LSTM Эпоха {epoch}: Потери = {loss.item():.4f}")

plt.plot(lstm_losses)
plt.title("LSTM Потери при обучении")
plt.xlabel("Эпоха")
plt.ylabel("Потери")
plt.grid()
plt.show()

lstm.eval()
with torch.no_grad():
    y_seq_pred = lstm(X_seq_test_tensor)
    _, y_seq_class = torch.max(y_seq_pred, 1)
    report_seq = classification_report(y_seq_test_tensor, y_seq_class, output_dict=True, zero_division=0)
    print("\n________Отчёт по классификации LSTM: 7-дневный прогноз________")
    for label in ["0", "1"]:
        print(f"{labels_map[label]}:")
        print(f"  Точность:  {report_seq[label]['precision']:.2f}")
        print(f"  Полнота:   {report_seq[label]['recall']:.2f}")
        print(f"  F-мера:    {report_seq[label]['f1-score']:.2f}")
    print(f"Общая точность: {report_seq['accuracy']:.2f}")

    cm_lstm = confusion_matrix(y_seq_test_tensor, y_seq_class)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm_lstm, annot=True, fmt="d", cmap="Greens", xticklabels=["Без дождя", "Дождь"],
                yticklabels=["Без дождя", "Дождь"])
    plt.xlabel("Предсказание")
    plt.ylabel("Истинное значение")
    plt.title("Матрица ошибок")
    plt.show()
