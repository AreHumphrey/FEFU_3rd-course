import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

digits = load_digits()
X, y = digits.data, digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=501, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f'Точность модели: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1-score: {f1:.4f}')

print("\nПараметры модели (веса и смещения первого слоя):")
print(f"Веса первого слоя:\n{model.coefs_[0]}")
print(f"Смещения первого слоя:\n{model.intercepts_[0]}")

# График потерь
plt.plot(model.loss_curve_)
plt.title("График функции потерь")
plt.xlabel("Эпохи")
plt.ylabel("Потери")
plt.grid(True)
plt.show()

fig, axes = plt.subplots(2, 5, figsize=(10, 5))
axes = axes.flatten()

for i in range(10):
    img = X_test[i].reshape(8, 8)
    axes[i].imshow(img, cmap='gray')
    axes[i].set_title(f"Предсказано: {y_pred[i]}\nИстинное: {y_test[i]}")
    axes[i].axis('off')

plt.tight_layout()
plt.show()
