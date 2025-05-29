import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data.csv")

df.drop(columns=["id"], inplace=True)

df.dropna(inplace=True)

y = df["Price"]
X = df.drop(columns=["Price"])

categorical_features = [
    "Brand", "Material", "Size",
    "Laptop Compartment", "Waterproof",
    "Style", "Color"
]
numeric_features = ["Compartments", "Weight Capacity (kg)"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

pipeline = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", RandomForestRegressor(n_estimators=100, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"средняя абсолютная ошибка: {mae:.2f}")
print(f"среднеквадратичная ошибка: {mse:.2f}")
print(f"корень из среднеквадратичной:             {rmse:.2f}")
print(f"коэффициент детерминации:  {r2:.2f}")

plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.7)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label='Идеальное совпадение')
plt.xlabel("Фактическая цена")
plt.ylabel("Предсказанная цена")
plt.title("Фактическая vs Предсказанная цена рюкзака")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
