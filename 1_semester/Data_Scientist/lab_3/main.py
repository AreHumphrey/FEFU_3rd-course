import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('Titanic-Dataset.csv')

data.rename(columns={
    'PassengerId': 'ИдентификаторПассажира',
    'Survived': 'Выжил',
    'Pclass': 'КлассПассажира',
    'Name': 'Имя',
    'Sex': 'Пол',
    'Age': 'Возраст',
    'SibSp': 'БратьяСестрыСупруги',
    'Parch': 'РодителиДети',
    'Ticket': 'Билет',
    'Fare': 'Плата',
    'Cabin': 'Каюта',
    'Embarked': 'ПортПосадки'
}, inplace=True)

print(data.info())
print(data.describe())
print(data.isnull().sum())

plt.figure(figsize=(10, 6))
sns.heatmap(data.isnull(), cbar=False, cmap='viridis')
plt.title("Пропущенные значения в датасете")
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(x='Выжил', data=data)
plt.title("Распределение выживших пассажиров")
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='КлассПассажира', y='Возраст', data=data)
plt.title("Возраст пассажиров в зависимости от класса")
plt.show()

data.drop(['Имя', 'Билет', 'Каюта'], axis=1, inplace=True)

label_encoder = LabelEncoder()
data['Пол'] = label_encoder.fit_transform(data['Пол'])
data = pd.get_dummies(data, columns=['ПортПосадки'], drop_first=True)

data['Возраст'].fillna(data['Возраст'].median(), inplace=True)
data['Плата'].fillna(data['Плата'].median(), inplace=True)

print("Пропущенные значения после обработки:")
print(data.isnull().sum())

y = data['Выжил']
X = data.drop('Выжил', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)
linear_mse = mean_squared_error(y_test, y_pred_linear)
print(f'Среднеквадратичная ошибка линейной регрессии: {linear_mse}')

logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train, y_train)
y_pred_logistic = logistic_model.predict(X_test)
logistic_accuracy = accuracy_score(y_test, y_pred_logistic)
print(f'Точность логистической регрессии: {logistic_accuracy}')
print("Отчет классификации логистической регрессии:\n", classification_report(y_test, y_pred_logistic))

lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X_train, y_train)
y_pred_lasso = lasso_model.predict(X_test)
lasso_mse = mean_squared_error(y_test, y_pred_lasso)
print(f'Среднеквадратичная ошибка Лассо-регрессии: {lasso_mse}')

y_pred_lasso_binary = np.where(y_pred_lasso >= 0.5, 1, 0)
lasso_accuracy = accuracy_score(y_test, y_pred_lasso_binary)
print(f'Точность Лассо-регрессии: {lasso_accuracy}')

print("Обработанный датасет:")
print(data.head())
