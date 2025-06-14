import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score, f1_score

file_path = 'steel_plates_faults_original_dataset.csv'
df = pd.read_csv(file_path)

target_columns = ['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']
X = df.drop(target_columns, axis=1)
y = df[target_columns]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MultiOutputClassifier(RandomForestClassifier(random_state=42))
model.fit(X_train, y_train)

preds = model.predict(X_test)

print("Отчёт классификации по каждому типу дефекта:\n")

for i, col in enumerate(target_columns):
    print(f"=== {col} ===")
    report = classification_report(y_test.iloc[:, i], preds[:, i], zero_division=0)
    print(report.replace('precision', 'Точность')
          .replace('recall', 'Полнота')
          .replace('f1-score', 'F1-мера')
          .replace('support', 'Поддержка'))

for i, col in enumerate(target_columns):
    cm = confusion_matrix(y_test.iloc[:, i], preds[:, i])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[f'Не {col}', f'{col}'])
    disp.plot(cmap='Blues')
    plt.title(f'Матрица ошибок — {col}')
    plt.show()

results = []
for i, col in enumerate(target_columns):
    results.append({
        'Defect': col,
        'Precision': precision_score(y_test.iloc[:, i], preds[:, i], zero_division=0),
        'Recall': recall_score(y_test.iloc[:, i], preds[:, i], zero_division=0),
        'F1-Score': f1_score(y_test.iloc[:, i], preds[:, i], zero_division=0)
    })

results_df = pd.DataFrame(results).round(2)
print("\n=== Сводные метрики по дефектам ===")
print(results_df.to_string(index=False))
