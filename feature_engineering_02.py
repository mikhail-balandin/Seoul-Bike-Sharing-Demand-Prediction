import pandas as pd
from eda_01 import df

# Обучаем модель только на Functioning Day = yes
# для Functioning Day = No прогнозируем 0

# После фильтрации столбец Functioning Day станет константой, поэтому его можно удалить

# Признаки из поля Date
# DayOfWeek - Holiday не заменяет день недели. Как вариант - использовать One Hot Encoding

# IsWeekend - можно вместо DayOfWeek использовать такое компактное бинарное поле
# Month - месяц может быть более информативным, чем Seasons

# Year - константа, тк у нас датасет только за 1 год. Дропаем

# Hour - будем использовать one hot encoding с 24 категориями часа



# Итоги:
# Для Seasons применим one hot encoding
# Holiday - преобразуем в бинарный признак и переназовем "IsHoliday"
# Functioning Day - дропаем

# погодные признаки пока удалять не будем, тк связь может быть и нелинейной
# Dew point temperature и Temperature сильно коррелируют, ради эксперимента стоит попробовать удалить Dew point temperature

TARGET_COLUMN = "Rented Bike Count"

def build_features(data: pd.DataFrame) -> pd.DataFrame:
    """Создаёт признаки для обучения моделей."""
    features = data.copy()

    # Календарные признаки
    features["DayOfWeek"] = features["Date"].dt.day_of_week
    features["IsHoliday"] = (
        features["Holiday"] == "Holiday"
    ).astype("int8")

    # Удаляем исходные и временно неиспользуемые столбцы
    features = features.drop(
        columns=[
            "Date",
            "Holiday",
            "Functioning Day",
            "Dew point temperature(°C)",
        ]
    )

    # Кодируем категориальные признаки
    features = pd.get_dummies(
        features,
        columns=["Hour", "Seasons", "DayOfWeek"],
        drop_first=True,
        dtype="int8",
    )

    return features

# Обучаем модель только на рабочих часах проката
operating_df = df.loc[
    df["Functioning Day"] == "Yes"
].copy()

feature_df = build_features(operating_df)

# Отделяем признаки от целевой переменной
X = feature_df.drop(columns=[TARGET_COLUMN])
y = feature_df[TARGET_COLUMN]

if __name__ == "__main__":
    print(f"Operating data: {operating_df.shape}")
    print(f"X: {X.shape}")
    print(f"y: {y.shape}")
    print(f"Missing values in X: {X.isna().sum().sum()}")
    print("\nFeature columns:")
    print(X.columns.tolist())