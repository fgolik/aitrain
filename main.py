import pandas as pd

# Загружаем датасет
df = pd.read_csv("train.csv")  # Убедись, что файл train.csv в папке проекта

# Вывод первых строк
print(df.head())
