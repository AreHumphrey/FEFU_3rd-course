import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

data_path = "spaceship_titanic.csv"
df = pd.read_csv(data_path)

print("Информация о данных:")
print(df.info())
print("\nПример данных:")
print(df.head())

df.drop(["PassengerId", "Name", "Cabin"], axis=1, inplace=True)

df["CryoSleep"] = df["CryoSleep"].fillna(False).astype(bool)

for col in ["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]:
    df[col] = df[col].fillna(df[col].median())

df = pd.get_dummies(df, columns=["HomePlanet", "Destination"], drop_first=True)

processed_data_path = "обработанные_данные_spaceship_titanic.csv"

df.to_csv(processed_data_path, index=False, encoding="utf-8")
print(f"Обработанные данные сохранены в файл: {processed_data_path}")

X = df.drop("Transported", axis=1)
y = df["Transported"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

xgb_model = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
xgb_model.fit(X_train, y_train)
xgb_predictions = xgb_model.predict(X_test)

rf_report = classification_report(y_test, rf_predictions, target_names=["Не транспортирован", "Транспортирован"],
                                  zero_division=0)
xgb_report = classification_report(y_test, xgb_predictions, target_names=["Не транспортирован", "Транспортирован"],
                                   zero_division=0)

results_path = "результаты_spaceship_titanic.txt"
with open(results_path, "w", encoding="utf-8") as file:
    file.write("Отчет классификации Random Forest:\n")
    file.write(rf_report)
    file.write("\nОтчет классификации XGBoost:\n")
    file.write(xgb_report)

feature_importance = rf_model.feature_importances_
plt.figure(figsize=(10, 6))
plt.bar(X.columns, feature_importance)
plt.xticks(rotation=45, ha="right")
plt.title("Важность признаков (Random Forest)")
plt.tight_layout()
plt.savefig("важность_признаков_rf.png")
plt.show()

print(f"Результаты сохранены в файл: {results_path}")
print("График важности признаков сохранен как: важность_признаков_rf.png")
