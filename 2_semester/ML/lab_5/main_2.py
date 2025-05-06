import kagglehub
import os
import pandas as pd
import numpy as np
from sklearn.neural_network import BernoulliRBM
import warnings

warnings.filterwarnings("ignore")

path = kagglehub.dataset_download("rounakbanik/the-movies-dataset")
print("Path to dataset files:", path)

ratings_path = os.path.join(path, "ratings_small.csv")
movies_path = os.path.join(path, "movies_metadata.csv")

ratings = pd.read_csv(ratings_path)
movies = pd.read_csv(movies_path, low_memory=False)

user_movie_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
X = user_movie_matrix.values
movie_ids = user_movie_matrix.columns

X_binary = (X > 0).astype(float)

rbm = BernoulliRBM(n_components=100, learning_rate=0.01, n_iter=20, verbose=True, random_state=42)
rbm.fit(X_binary)

reconstructed = rbm.gibbs(X_binary)

user_id = 10
user_index = user_id - 1
user_ratings = X[user_index]
user_reconstructed = reconstructed[user_index]

unseen = np.where(user_ratings == 0)[0]
recommended_indices = unseen[np.argsort(user_reconstructed[unseen])[::-1][:10]]
recommended_movie_ids = movie_ids[recommended_indices].astype(str)

movies = movies.dropna(subset=['id', 'title'])
movies['id'] = movies['id'].astype(str)
recommended_titles = movies[movies['id'].isin(recommended_movie_ids)]['title'].unique()

print(f"\nРекомендованные фильмы для пользователя {user_id}:\n")
for i, title in enumerate(recommended_titles, 1):
    print(f"{i}. {title}")
