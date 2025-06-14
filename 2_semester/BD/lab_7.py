import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

print("Загрузка данных...")
df = pd.read_csv('spotifydataset.csv')

print("Предобработка данных...")
df['explicit'] = df['explicit'].astype(int)
X = df.drop(
    columns=['artist_name', 'track_name', 'album_name', 'release_date', 'artist_url', 'genres', 'track_popularity'])
y = df['track_popularity']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Кластеризация с помощью K-Means...")
inertias = []
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 10), inertias, marker='o')
plt.title('Метод локтя для выбора числа кластеров')
plt.xlabel('Число кластеров')
plt.ylabel('Inertia')
plt.grid(True)
plt.show()

kmeans = KMeans(n_clusters=5, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

print("\nРаспределение по кластерам:")
print(df['cluster'].value_counts())

print("Прогнозирование популярности трека...")
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nMSE: {mse:.2f}")
print(f"R²: {r2:.2f}")

plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_test, y=y_pred)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')
plt.title('Фактическая vs Предсказанная популярность')
plt.xlabel('Фактическая popular')
plt.ylabel('Предсказанная popular')
plt.grid(True)
plt.show()

print("Важность признаков...")
features = X.columns
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Важность признаков")
sns.barplot(x=features[indices], y=importances[indices])
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


def recommend_similar_tracks(track_index, n_recommendations=5):
    track_features = X_scaled[track_index].reshape(1, -1)
    distances = np.linalg.norm(X_scaled - track_features, axis=1)
    nearest_indices = np.argsort(distances)[1:n_recommendations + 1]
    return df.iloc[nearest_indices][['track_name', 'artist_name', 'track_popularity']]


print("\nРекомендации для трека:", df.iloc[0]['track_name'])
print(recommend_similar_tracks(0))

df.to_csv('spotify_with_clusters_and_predictions.csv', index=False)
