import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

df = pd.read_excel('data.xlsx')


for col in ['x1', 'x2', 'x3', 'x4', 'x5']:
    df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'\D', '', regex=True), errors='coerce')

for col in ['x1', 'x2', 'x3', 'x4', 'x5']:
    df[col] = df[col].fillna(df[col].median())

def delete_outliers(df, feature):
    _x = df.boxplot(column=feature)
    plt.close('all')

    whiskers = []
    for line in _x.lines:
        if line.get_linestyle() == '-':
            whiskers.append(line.get_ydata())

    whiskers = whiskers[1:3]
    lower_whisker = whiskers[0][1]
    upper_whisker = whiskers[1][1]

    median = np.median(df[feature])
    df[feature] = np.where((df[feature] > upper_whisker) | (df[feature] < lower_whisker), median, df[feature])

features = ['x1', 'x2', 'x3', 'x4', 'x5']
for feature in features:
    delete_outliers(df, feature)

output_file = os.path.join(os.getcwd(), 'processed_data.xlsx')
print(f"Файл будет сохранен по пути: {output_file}")


plt.boxplot([df[feature] for feature in features], whis=(0, 100))
plt.xticks(range(1, len(features) + 1), features)
plt.show()
