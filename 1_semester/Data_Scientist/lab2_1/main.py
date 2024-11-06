import pandas as pd
import numpy as np
import re
from collections import Counter

with open('data.txt', 'r') as file:
    data = [line.strip() for line in file if line.strip() != '']

columns = ['Text']
df = pd.DataFrame(data, columns=columns)

# Замена гадких слов
bad_values = ['N/A', 'na', 'NaN', None, 'null', '-', '?', 'undefined']
df.replace(bad_values, np.nan, inplace=True)
print("\nЭтап 1: После замены плохих значений:\n")
print(df.head(10).to_string(index=False))

# Очистка текста от нежелательных словечек, приведение к единому формату
df['Text'] = df['Text'].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', x).lower() if isinstance(x, str) else x)
print("\nЭтап 2: После очистки текста:\n")
print(df.head(10).to_string(index=False))

# Приведение текста в числа
all_words = [word for text in df['Text'] if isinstance(text, str) for word in text.split()]
word_counts = Counter(all_words)
most_common_words = [word for word, _ in word_counts.most_common()]
word_to_num = {word: idx + 1 for idx, word in enumerate(most_common_words)}

# Преобразование текста в числовой формат на основе общего словаря
df['Numeric_Text'] = df['Text'].apply(lambda x: [word_to_num[word] for word in x.split()] if isinstance(x, str) else [])

df['Numeric_Text'] = df['Numeric_Text'].apply(lambda x: ' '.join(map(str, x)))
print("\nЭтап 3: После приведения текста в числовой формат:\n")
print(df.head(10).to_string(index=False))

# Приведение длины текста и количества слов
df['Text_Length'] = df['Text'].apply(lambda x: len(x) if isinstance(x, str) else 0)
df['Word_Count'] = df['Text'].apply(lambda x: len(x.split()) if isinstance(x, str) else 0)


# Удаление выбросов на основе длины текста и количества слов
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]


for column in ['Text_Length', 'Word_Count']:
    df = remove_outliers(df, column)

print("\nЭтап 4: После удаления выбросов:\n")
print(df.head(10).to_string(index=False))

# Сохраняем очищенные данные в новый файл
df.to_csv('cleaned_data.csv', index=False)
