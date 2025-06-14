from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import torch
import torch.nn as nn
import random

dataset = load_dataset("ag_news", split="train[:5000]")

texts = [item['text'] for item in dataset]
labels = [item['label'] for item in dataset]

bow_vectorizer = CountVectorizer(max_features=1000)
X_bow = bow_vectorizer.fit_transform(texts).toarray()

ngram_vectorizer = CountVectorizer(ngram_range=(2, 2), max_features=1000)
X_ngram = ngram_vectorizer.fit_transform(texts).toarray()


class TransformerTextGen(nn.Module):
    def __init__(self, input_dim, emb_dim=128, nhead=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, emb_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(emb_dim, input_dim)

    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)
        x = self.transformer(x)
        x = self.decoder(x).squeeze(1)
        return x


X_train, X_test = train_test_split(X_bow, test_size=0.2, random_state=42)

model = TransformerTextGen(input_dim=X_bow.shape[1])
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

accuracy_per_epoch = []

for epoch in range(10):
    model.train()
    optimizer.zero_grad()
    output = model(X_train_tensor)
    loss = criterion(output, X_train_tensor)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = model(X_test_tensor)
        true = (X_test_tensor.numpy() > 0).astype(int)
        pred = (preds.numpy() > 0.5).astype(int)
        acc = accuracy_score(true.flatten(), pred.flatten())
        accuracy_per_epoch.append(acc)

    print(f"Эпоха {epoch + 1}: Потери = {loss.item():.4f} | Точность = {acc:.4f}")


precision = precision_score(true.flatten(), pred.flatten(), average='macro', zero_division=0)
recall = recall_score(true.flatten(), pred.flatten(), average='macro', zero_division=0)

print(f"\nИтоговая точность: {accuracy_per_epoch[-1]:.4f}")
print(f"Итоговая точность (Precision): {precision:.4f}")
print(f"Итоговая полнота (Recall): {recall:.4f}")

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), accuracy_per_epoch, marker='o', label='Точность')
plt.title("Точность модели по эпохам (BoW)")
plt.xlabel("Эпоха")
plt.ylabel("Точность")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


def generate_text(model, vectorizer, input_vector=None, top_k=5):
    model.eval()
    with torch.no_grad():
        if input_vector is None:
            idx = random.randint(0, len(X_test_tensor))
            input_tensor = X_test_tensor[idx].unsqueeze(0)
        else:
            input_tensor = torch.tensor(input_vector, dtype=torch.float32).unsqueeze(0)

        output = model(input_tensor)
        probs = torch.sigmoid(output).squeeze().numpy()

    word_indices = probs.argsort()[-top_k:][::-1]
    words = vectorizer.get_feature_names_out()
    predicted_words = [words[i] for i in word_indices]
    return ' '.join(predicted_words)

print("\n--- Примеры сгенерированных фраз ---")
for _ in range(5):
    generated = generate_text(model, bow_vectorizer)
    print(f"Сгенерировано: {generated}")
