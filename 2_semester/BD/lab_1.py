import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import re
import random


text = """
Однажды утром Грегор Замза проснулся после беспокойного сна и обнаружил, что он превратился в огромное насекомое. 
Он лежал на спине, твёрдой как панцирь, и, приподняв немного голову, увидел свой коричневый выпуклый живот, разделённый на дугообразные твёрдые секции. 
Одеяло, едва державшееся на вершине живота, вот-вот должно было сползти окончательно. 
Его многочисленные, жалкие по сравнению с телом ножки беспомощно мелькали перед глазами.
\"Что со мной случилось?\" — подумал он. Это не был сон. Его комната, настоящая, вполне обычная комната, разве что немного мала, покоилась спокойно между четырьмя знакомыми стенами.
На столе разложены образцы тканей — Замза был коммивояжёром — и с ящика над столом свисал вырезанный из иллюстрированного журнала недавно вставленный в красивую позолоченную рамку портрет дамы.
Её изображение было заклеено плотной бумагой, и только вырезанные по трафарету глаза и рот были видны. Грегор взглянул в окно. За окном было пасмурно, слышался лёгкий дождь.
\"Какой унылый день\", — подумал он. Потом его глаза скользнули по будильнику. \"Боже!\" — подумал он. \"Уже шесть с половиной, а я всё ещё не встал!\"
Он был привык вставать рано, потому что работа коммивояжёра требует строгости и пунктуальности.
"""


def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text.split()

tokens = preprocess(text)
vocab = sorted(set(tokens))
word_to_ix = {word: i for i, word in enumerate(vocab)}
ix_to_word = {i: word for word, i in word_to_ix.items()}

n = 2
grams = [(tokens[i:i+n], tokens[i+n]) for i in range(len(tokens)-n)]
def prepare_seq(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

X = [prepare_seq(x, word_to_ix) for x, _ in grams]
y = [word_to_ix[label] for _, label in grams]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

def bag_of_words(tokens, word_to_ix):
    vectors = []
    labels = []
    for i in range(2, len(tokens)):
        vec = [0] * len(word_to_ix)
        for word in tokens[i-2:i]:
            vec[word_to_ix[word]] += 1
        vectors.append(torch.tensor(vec, dtype=torch.float32))
        labels.append(word_to_ix[tokens[i]])
    return vectors, labels

X_bow, y_bow = bag_of_words(tokens, word_to_ix)
X_train_bow, X_test_bow, y_train_bow, y_test_bow = train_test_split(X_bow, y_bow, test_size=0.3, random_state=42)

def batchify(X, y, batch_size=4):
    zipped = list(zip(X, y))
    random.shuffle(zipped)
    batches = [zipped[i:i+batch_size] for i in range(0, len(zipped), batch_size)]
    return batches

class RNNModel(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.RNN(emb_dim, hid_dim, batch_first=True)
        self.fc = nn.Linear(hid_dim, vocab_size)
    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])

class GRUModel(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.gru = nn.GRU(emb_dim, hid_dim, batch_first=True)
        self.fc = nn.Linear(hid_dim, vocab_size)
    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hid_dim, batch_first=True)
        self.fc = nn.Linear(hid_dim, vocab_size)
    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

class BoWModel(nn.Module):
    def __init__(self, vocab_size, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(vocab_size, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, vocab_size)
    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)


def train_model(model, batches, epochs=100):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in batches:
            x_batch = torch.stack([x for x, _ in batch])
            y_batch = torch.tensor([y for _, y in batch])
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = loss_fn(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Loss = {total_loss:.4f}")


def evaluate_model(model, X_test, y_test):
    model.eval()
    y_pred = []
    with torch.no_grad():
        for x in X_test:
            x = x.unsqueeze(0)
            output = model(x)
            probs = torch.softmax(output, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            y_pred.append(pred)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
    return acc, prec, rec

vocab_size = len(vocab)
embedding_dim = 10
hidden_dim = 20

print("\n--- RNN Training ---")
rnn = RNNModel(vocab_size, embedding_dim, hidden_dim)
train_model(rnn, batchify(X_train, y_train), 100)
acc, prec, rec = evaluate_model(rnn, X_test, y_test)
print(f"\n Метрики RNN: Accuracy = {acc:.2f}, Precision = {prec:.2f}, Recall = {rec:.2f}")

print("\n--- GRU Training ---")
gru = GRUModel(vocab_size, embedding_dim, hidden_dim)
train_model(gru, batchify(X_train, y_train), 100)
acc, prec, rec = evaluate_model(gru, X_test, y_test)
print(f"\n Метрики GRU: Accuracy = {acc:.2f}, Precision = {prec:.2f}, Recall = {rec:.2f}")

print("\n--- LSTM Training ---")
lstm = LSTMModel(vocab_size, embedding_dim, hidden_dim)
train_model(lstm, batchify(X_train, y_train), 100)
acc, prec, rec = evaluate_model(lstm, X_test, y_test)
print(f"\nМетрики LSTM: Accuracy = {acc:.2f}, Precision = {prec:.2f}, Recall = {rec:.2f}")

print("\n--- BoW Training ---")
bow = BoWModel(vocab_size, hidden_dim)
train_model(bow, batchify(X_train_bow, y_train_bow), 100)
acc, prec, rec = evaluate_model(bow, X_test_bow, y_test_bow)
print(f"\n Метрики BoW: Accuracy = {acc:.2f}, Precision = {prec:.2f}, Recall = {rec:.2f}")


def print_predictions(model, X_test, y_test, model_name):
    print(f"\nПримеры предсказаний для модели {model_name}:")
    model.eval()
    with torch.no_grad():
        for i in range(5):
            x = X_test[i].unsqueeze(0)
            output = model(x)
            probs = torch.softmax(output, dim=1).squeeze()
            top3 = torch.topk(probs, 3)
            top_words = [ix_to_word[idx.item()] for idx in top3.indices]
            real = ix_to_word[y_test[i]]
            context = " ".join([ix_to_word[idx.item()] for idx in X_test[i]])
            print(f"Контекст: {context:<30} | Ожидаемое слово: {real:<10} | Предсказано: {top_words[0]}")
            print(f"  Наиболее вероятные варианты: {', '.join(top_words)}\n")

print_predictions(rnn, X_test, y_test, "RNN")
print_predictions(gru, X_test, y_test, "GRU")
print_predictions(lstm, X_test, y_test, "LSTM")

def print_predictions_bow(model, X_test, y_test, model_name):
    print(f"\nПримеры предсказаний для модели {model_name} (Bag of Words):")
    model.eval()
    with torch.no_grad():
        for i in range(5):
            x = X_test[i].unsqueeze(0)
            output = model(x)
            probs = torch.softmax(output, dim=1).squeeze()
            top3 = torch.topk(probs, 3)
            top_words = [ix_to_word[idx.item()] for idx in top3.indices]
            real = ix_to_word[y_test[i]]
            print(f"Контекст (вектор признаков):       | Ожидаемое слово: {real:<10} | Предсказано: {top_words[0]}")
            print(f"  Наиболее вероятные варианты: {', '.join(top_words)}\n")

print_predictions_bow(bow, X_test_bow, y_test_bow, "BoW")


def train_model(model, batches, epochs=100):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_history = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in batches:
            x_batch = torch.stack([x for x, _ in batch])
            y_batch = torch.tensor([y for _, y in batch])
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = loss_fn(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        loss_history.append(total_loss)
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Loss = {total_loss:.4f}")
    return loss_history

import matplotlib.pyplot as plt

losses = {}

print("\n--- RNN Training ---")
rnn = RNNModel(vocab_size, embedding_dim, hidden_dim)
losses["RNN"] = train_model(rnn, batchify(X_train, y_train), 100)

print("\n--- GRU Training ---")
gru = GRUModel(vocab_size, embedding_dim, hidden_dim)
losses["GRU"] = train_model(gru, batchify(X_train, y_train), 100)

print("\n--- LSTM Training ---")
lstm = LSTMModel(vocab_size, embedding_dim, hidden_dim)
losses["LSTM"] = train_model(lstm, batchify(X_train, y_train), 100)

print("\n--- BoW Training ---")
bow = BoWModel(vocab_size, hidden_dim)
losses["BoW"] = train_model(bow, batchify(X_train_bow, y_train_bow), 100)

plt.figure(figsize=(10, 6))
for name, loss_vals in losses.items():
    plt.plot(loss_vals, label=name)
plt.title("График потерь по эпохам")
plt.xlabel("Эпоха")
plt.ylabel("Суммарная потеря")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
