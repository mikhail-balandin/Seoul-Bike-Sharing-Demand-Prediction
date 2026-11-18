from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
FIGURES_DIR = BASE_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.float_format', '{:.2f}'.format)

sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['figure.figsize'] = (12, 6)

df = pd.read_csv(BASE_DIR / "SeoulBikeData.csv", encoding="latin1")

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
fig.savefig(FIGURES_DIR / "01_distributions.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close(fig)

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
fig.savefig(FIGURES_DIR / "02_rentals.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close(fig)

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
fig.savefig(FIGURES_DIR / "03_boxplots.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close(fig)

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
fig.savefig(FIGURES_DIR / "04_categories.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close(fig)

# Матрица корреляций. Автор кода: Никита Саралашвили.
# Корреляции считаются по числовым столбцам; исходный df не меняется.
numeric_df = df.select_dtypes(include="number")
corr = numeric_df.corr(method="pearson")

labels = {
    "Rented Bike Count": "Прокаты за час",
    "Hour": "Час суток",
    "Humidity(%)": "Влажность",
    "Wind speed (m/s)": "Скорость ветра",
    "Visibility (10m)": "Видимость",
    "Solar Radiation (MJ/m2)": "Солнечная радиация",
    "Rainfall(mm)": "Дождь",
    "Snowfall (cm)": "Снег",
}
for column in corr.columns:
    if column.startswith("Temperature("):
        labels[column] = "Температура"
    elif column.startswith("Dew point temperature("):
        labels[column] = "Точка росы"

corr_named = corr.rename(index=labels, columns=labels)
fig, ax = plt.subplots(figsize=(13, 11))
sns.heatmap(
    corr_named, annot=True, fmt=".2f", cmap="coolwarm",
    vmin=-1, vmax=1, center=0, square=True, linewidths=0.5,
    annot_kws={"size": 10},
    cbar_kws={"label": "Коэффициент корреляции Пирсона"}, ax=ax,
)
ax.set_title("Связи числовых показателей: все 8760 часов", pad=18)
plt.xticks(rotation=40, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
fig.savefig(FIGURES_DIR / "05_correlations.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close(fig)

# Связи со спросом, от наиболее заметных по модулю к наименее заметным.
demand_corr = corr_named["Прокаты за час"].drop("Прокаты за час")
demand_corr = demand_corr.reindex(demand_corr.abs().sort_values(ascending=False).index)
print(demand_corr.round(3).to_frame("Корреляция с количеством прокатов").to_string())



def plot_pair(x, y, xlabel, ylabel, title, filename):
    fig, ax = plt.subplots(figsize=(9, 5.5), layout="constrained")
    ax.scatter(x, y, s=9, color="#15668b", alpha=0.14, linewidths=0)
    slope, intercept = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 200)
    ys = slope * xs + intercept
    visible = (ys >= y.min()) & (ys <= y.max())
    ax.plot(xs[visible], ys[visible], color="#bb411d", lw=3, label="Линейная тенденция")
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.legend()
    fig.savefig(FIGURES_DIR / filename, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close(fig)

plot_pair(df["Temperature(°C)"], df["Rented Bike Count"],
          "Температура, °C", "Прокаты за час", "Температура и прокаты",
          "06_temperature_rentals.png")
plot_pair(df["Humidity(%)"], df["Visibility (10m)"] / 100,
          "Влажность, %", "Видимость, км", "Влажность и видимость",
          "07_humidity_visibility.png")

from PIL import Image

panels = [
    ("01_distributions.png", "Распределения числовых показателей"),
    ("03_boxplots.png", "Разброс и крайние значения"),
    ("02_rentals.png", "Распределение прокатов"),
    ("04_categories.png", "Сезоны, праздники и работа проката"),
    ("05_correlations.png", "Корреляции числовых показателей"),
    ("06_temperature_rentals.png", "Температура и прокаты"),
    ("07_humidity_visibility.png", "Влажность и видимость"),
]
fig = plt.figure(figsize=(18, 27), layout="constrained")
grid = fig.add_gridspec(4, 2, height_ratios=[9, 3.3, 8, 5])
positions = [grid[0, 0], grid[0, 1], grid[1, 0], grid[1, 1],
             grid[2, :], grid[3, 0], grid[3, 1]]
for (filename, title), position in zip(panels, positions):
    ax = fig.add_subplot(position)
    with Image.open(FIGURES_DIR / filename) as panel:
        panel.thumbnail((1900, 1750), Image.Resampling.LANCZOS)
        ax.imshow(panel)
    ax.set_title(title, fontsize=14)
    ax.axis("off")
fig.suptitle("Прокат велосипедов в Сеуле | 8760 часовых наблюдений", fontsize=22)
fig.savefig(FIGURES_DIR / "00_overview.png", dpi=180, facecolor="white")
plt.close(fig)
