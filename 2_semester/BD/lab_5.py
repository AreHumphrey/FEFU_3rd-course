import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

file_path = 'depression_data.csv'
df = pd.read_csv(file_path)


df.drop(['id', 'City'], axis=1, inplace=True, errors='ignore')

def encode_sleep_duration(s):
    if s == 'Less than 5 hours':
        return 4.5
    elif s == '5-6 hours':
        return 5.5
    elif s == '7-8 hours':
        return 7.5
    elif s == 'More than 8 hours':
        return 9
    else:
        return float('nan')

df['Sleep Duration'] = df['Sleep Duration'].apply(encode_sleep_duration)

binary_map = {'Yes': 1, 'No': 0}
if 'Have you ever had suicidal thoughts ?' in df.columns:
    df['Have you ever had suicidal thoughts ?'] = df['Have you ever had suicidal thoughts ?'].map(binary_map)
if 'Family History of Mental Illness' in df.columns:
    df['Family History of Mental Illness'] = df['Family History of Mental Illness'].map(binary_map)

X = df.drop('Depression', axis=1, errors='ignore')
y = df['Depression']

categorical_features = X.select_dtypes(include='object').columns
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]), numerical_features),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ])

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model.fit(X_train, y_train)

preds = model.predict(X_test)
probs = model.predict_proba(X_test)[:, 1]

print("\nОтчёт классификации:")
print(classification_report(y_test, preds))

cm = confusion_matrix(y_test, preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Нет депрессии', 'Депрессия'], yticklabels=['Нет депрессии', 'Депрессия'])
plt.xlabel('Предсказанные значения')
plt.ylabel('Фактические значения')
plt.title('Матрица ошибок')
plt.show()

fpr, tpr, _ = roc_curve(y_test, probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC-кривая (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-кривая')
plt.legend(loc="lower right")
plt.show()

feature_names = numerical_features.tolist() + \
                 model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['encoder'].get_feature_names_out(categorical_features).tolist()

importances = model.named_steps['classifier'].feature_importances_
feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=feat_imp.head(10), y=feat_imp.head(10).index)
plt.title("Топ 10 важных признаков")
plt.tight_layout()
plt.show()
