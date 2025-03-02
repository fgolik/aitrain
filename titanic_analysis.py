import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Загружаем датасет
df = pd.read_csv("train.csv")

# 1. Обработка пропущенных значений
# Проверяем, где есть пропущенные значения
print("Перед заполнением пропущенных значений:")
print(df.isnull().sum())

# Заменяем пропущенные значения в столбце "Age" медианой
df["Age"] = df["Age"].fillna(df["Age"].median())

# Заменяем пропущенные значения в столбце "HomePlanet" модой
df["HomePlanet"] = df["HomePlanet"].fillna(df["HomePlanet"].mode()[0])

# Заменяем пропущенные значения в столбце "Cabin" на "Unknown"
df["Cabin"] = df["Cabin"].fillna("Unknown")

# Заменяем пропущенные значения в столбце "VIP" средним значением
df["VIP"] = df["VIP"].fillna(df["VIP"].mean())

# Проверяем, сколько пропущенных значений осталось
print("\nПосле заполнения пропущенных значений:")
print(df.isnull().sum())

# 2. Нормализация данных
# Инициализация нормализаторов
min_max_scaler = MinMaxScaler()  # Для MinMaxScaler
standard_scaler = StandardScaler()  # Для StandardScaler

# Применяем MinMaxScaler для столбца "Age"
df["Age"] = min_max_scaler.fit_transform(df[["Age"]])

# Или применяем StandardScaler для столбца "Age"
df["Age"] = standard_scaler.fit_transform(df[["Age"]])

# 3. One-hot encoding для столбца "FoodCourt"
df = pd.get_dummies(df, columns=["FoodCourt"], drop_first=True)

# Проверяем результат
print("\nДанные после нормализации и One-hot encoding:")
print(df.head())
df.to_csv("processed_train.csv", index=False)
