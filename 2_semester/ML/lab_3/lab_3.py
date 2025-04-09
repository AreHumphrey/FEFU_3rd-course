import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import os


(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train, y_test = to_categorical(y_train, 10), to_categorical(y_test, 10)

x_train, y_train = x_train[:3000], y_train[:3000]
x_test, y_test = x_test[:500], y_test[:500]


def create_model(dropout=False, batch_norm=False):
    model = Sequential([Flatten(input_shape=(28, 28))])
    for units in [64, 32]:
        model.add(Dense(units, activation='relu'))
        if batch_norm:
            model.add(BatchNormalization())
        if dropout:
            model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    return model


results = []


def plot_metrics(history, title=''):
    os.makedirs("plots", exist_ok=True)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].plot(history.history['accuracy'], label='Обучение')
    axs[0].plot(history.history['val_accuracy'], label='Тест')
    axs[0].set_title(f'{title} — Точность')
    axs[0].set_xlabel('Эпохи')
    axs[0].legend()

    axs[1].plot(history.history['loss'], label='Обучение')
    axs[1].plot(history.history['val_loss'], label='Тест')
    axs[1].set_title(f'{title} — Потери')
    axs[1].set_xlabel('Эпохи')
    axs[1].legend()

    plt.tight_layout()
    filename = f"plots/{title.replace(': ', '_').replace(' ', '_')}.png"
    plt.savefig(filename)
    plt.close()

    results.append({
        "Эксперимент": title,
        "Точность обучения": round(history.history['accuracy'][-1] * 100, 2),
        "Точность теста": round(history.history['val_accuracy'][-1] * 100, 2),
        "Потери обучения": round(history.history['loss'][-1], 4),
        "Потери теста": round(history.history['val_loss'][-1], 4),
        "Файл графика": filename
    })


def train_and_plot(model, optimizer, epochs, title):
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, verbose=0)
    plot_metrics(history, title)
    return history


for ep in [3, 5, 10]:
    train_and_plot(create_model(), Adam(), ep, f"Эпох: {ep}")

train_and_plot(create_model(), Adam(learning_rate=0.0001), 10, "LR = 0.0001")
train_and_plot(create_model(), Adam(learning_rate=0.01), 10, "LR = 0.01")
train_and_plot(create_model(dropout=False), Adam(), 10, "Без Dropout")
train_and_plot(create_model(dropout=True), Adam(), 10, "С Dropout")
train_and_plot(create_model(batch_norm=False), Adam(), 10, "Без BatchNorm")
train_and_plot(create_model(batch_norm=True), Adam(), 10, "С BatchNorm")

results_df = pd.DataFrame(results)
os.makedirs("results", exist_ok=True)
results_df.to_csv("results/результаты_экспериментов.csv", index=False)

print("\nТаблица итогов экспериментов:\n")
print(results_df.to_string(index=False))
