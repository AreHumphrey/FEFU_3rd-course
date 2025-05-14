import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Input
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

driver_types = {0: 'Осторожный', 1: 'Агрессивный', 2: 'Непредсказуемый'}


def generate_driver_behavior_data(num_samples=1500, sequence_length=20):
    X, y = [], []
    for _ in range(num_samples):
        behavior = np.random.choice([0, 1, 2])
        trip = []

        for _ in range(sequence_length):
            if behavior == 0:
                speed = np.random.uniform(30, 70)
                accel = np.random.uniform(0.0, 0.3)
                maneuver = 0 if np.random.rand() < 0.95 else 1
            elif behavior == 1:
                speed = np.random.uniform(80, 130)
                accel = np.random.uniform(0.7, 1.5)
                maneuver = 1 if np.random.rand() < 0.9 else 0
            else:
                speed = np.random.uniform(20, 130)
                accel = np.random.uniform(0.0, 1.5)
                maneuver = np.random.choice([0, 1])
            trip.append([speed, accel, maneuver])
        X.append(trip)
        y.append(behavior)
    return np.array(X), np.array(y)


X, y = generate_driver_behavior_data()
y_cat = to_categorical(y, num_classes=3)
X_train, X_test, y_train, y_test, y_raw_train, y_raw_test = train_test_split(
    X, y_cat, y, test_size=0.2, random_state=42
)

model = Sequential([
    Input(shape=(X.shape[1], X.shape[2])),
    SimpleRNN(32, activation='tanh'),
    Dense(3, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=15, batch_size=32, validation_split=0.1)

y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

print(classification_report(y_raw_test, y_pred, target_names=driver_types.values()))

conf_matrix = confusion_matrix(y_raw_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, xticklabels=driver_types.values(), yticklabels=driver_types.values(), fmt='d',
            cmap='Blues')
plt.title("Матрица ошибок: Классификация поведения водителя")
plt.xlabel("Предсказано")
plt.ylabel("Истинно")
plt.show()
model.save("driver_behavior_model.h5")
