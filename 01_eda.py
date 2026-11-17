import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.float_format', '{:.2f}'.format)

sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['figure.figsize'] = (12, 6)

df = pd.read_csv("./SeoulBikeData.csv", encoding="cp949")

# 1 EDA
# print(f'Df Shape: {df.shape}\n')
# print(df.head(10))

# df.info()

# print(df.describe(include='all').T)

# print('Columns:')
# print(df.columns.tolist())

# print('\nData types:')
# print(df.dtypes)

# print(f'Number of rows: {len(df)}')
# print(f'Number of columns: {len(df.columns)}')

# print(f'\nMemory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB')

# print(f'\nMissing values: {df.isna().sum().sum()}')
# print(f'Duplicated rows: {df.duplicated().sum()}')

# 2 Проверка качества данных

df['Date'] = pd.to_datetime(
    df['Date'],
    format='%d/%m/%Y'
)

print(df['Date'].dtype)
print(f"Start date: {df['Date'].min()}")
print(f"End date: {df['Date'].max()}")
print(f"Unique dates: {df['Date'].nunique()}")

# проверяем полноту временного ряда

daily_counts = df.groupby('Date')['Hour'].agg(
    ['count', 'nunique', 'min', 'max']
)

print(daily_counts.describe())

print("Days with != 24 observations:",
      (daily_counts['count'] != 24).sum())

print("Days with != 24 unique hours:",
      (daily_counts['nunique'] != 24).sum())

print("Missing dates:",
      365 - df['Date'].nunique())

# проверяем целевую переменную

print(df['Rented Bike Count'].describe())

print("\nZero rentals:",
      (df['Rented Bike Count'] == 0).sum())

print("\nNegative rentals:",
      (df['Rented Bike Count'] < 0).sum())

print("\nFunctioning Day distribution:")
print(df['Functioning Day'].value_counts())


print(
    df.groupby('Functioning Day')['Rented Bike Count']
    .agg(['count', 'mean', 'median', 'min', 'max', 'sum'])
)

categorical_columns = [
    'Seasons',
    'Holiday',
    'Functioning Day'
]

for col in categorical_columns:
    print(f'\n{col}')
    print(df[col].value_counts())
    print('Unique:', df[col].unique())

numeric_columns = df.select_dtypes(
    include=np.number
).columns

for col in numeric_columns:
    print(f'\n{col}')
    print(f'Min: {df[col].min()}')
    print(f'Max: {df[col].max()}')
    print(f'Negative values: {(df[col] < 0).sum()}')

# 3 визуальное исследование

# распределение числовых признаков

numeric_columns = df.select_dtypes(include=np.number).columns.tolist()

fig, axes = plt.subplots(4, 3, figsize=(18, 18))
axes = axes.flatten()

for i, col in enumerate(numeric_columns):
    sns.histplot(
        data=df,
        x=col,
        kde=True,
        ax=axes[i]
    )
    axes[i].set_title(f'Distribution of {col}')

for i in range(len(numeric_columns), len(axes)):
    axes[i].set_visible(False)

plt.tight_layout()
plt.show()

# целевая переменная

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

sns.histplot(
    df['Rented Bike Count'],
    kde=True,
    ax=axes[0]
)
axes[0].set_title('Rented Bike Count Distribution')

sns.histplot(
    np.log1p(df['Rented Bike Count']),
    kde=True,
    ax=axes[1]
)
axes[1].set_title('Log-transformed Rented Bike Count')

plt.tight_layout()
plt.show()

# поиск выбросов

fig, axes = plt.subplots(4, 3, figsize=(18, 18))
axes = axes.flatten()

for i, col in enumerate(numeric_columns):
    sns.boxplot(
        data=df,
        x=col,
        ax=axes[i]
    )
    axes[i].set_title(f'Boxplot of {col}')

for i in range(len(numeric_columns), len(axes)):
    axes[i].set_visible(False)

plt.tight_layout()
plt.show()

# анализ нулевых значений осадков
weather_columns = [
    'Rainfall(mm)',
    'Snowfall (cm)',
    'Solar Radiation (MJ/m2)'
]

for col in weather_columns:
    zero_count = (df[col] == 0).sum()
    zero_percent = zero_count / len(df) * 100

    print(
        f'{col}: {zero_count} zeros '
        f'({zero_percent:.2f}%)'
    )


# канализ категориальных переменных
categorical_columns = [
    'Seasons',
    'Holiday',
    'Functioning Day'
]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for i, col in enumerate(categorical_columns):
    sns.countplot(
        data=df,
        x=col,
        ax=axes[i]
    )

    axes[i].set_title(f'Distribution of {col}')
    axes[i].tick_params(axis='x', rotation=20)

plt.tight_layout()
plt.show()