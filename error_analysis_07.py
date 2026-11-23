# Разбор ошибок финальной HGB-модели для презентации.

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import HistGradientBoostingRegressor

from eda_01 import df
from feature_engineering_02 import X, y, operating_df
from modeling_utils import calculate_metrics, split_data

BASE_DIR = Path(__file__).resolve().parent
FIGURES_DIR = BASE_DIR / "figures"
REPORTS_DIR = BASE_DIR / "reports"

RANDOM_STATE = 42
MIN_GROUP_SIZE = 20

SEASON_LABELS = {
    "Winter": "Зима",
    "Spring": "Весна",
    "Summer": "Лето",
    "Autumn": "Осень",
}

WEATHER_LABELS = {
    "Dry": "Сухо",
    "Rain": "Дождь",
    "Snow": "Снег",
    "Rain + snow": "Дождь + снег",
}

HOLIDAY_LABELS = {
    "Regular": "Обычный",
    "Holiday": "Праздник",
}

DAY_TYPE_LABELS = {
    "Weekday": "Будни",
    "Weekend": "Выходные",
}

SEGMENT_GROUPS = [
    ("Hour", "Hour"),
    ("Season", "Seasons"),
    ("Day type", "Day type"),
    ("Holiday", "Holiday status"),
    ("Weather", "Weather"),
    ("Temperature", "Temperature band"),
    ("Actual demand", "Demand band"),
]


def fit_final_model():
    """Обучает финальную HGB-модель и возвращает прогнозы на test."""
    (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
    ) = split_data(X, y, random_state=RANDOM_STATE)

    X_development = pd.concat([X_train, X_validation])
    y_development = pd.concat([y_train, y_validation])

    model = HistGradientBoostingRegressor(
        max_iter=200,
        learning_rate=0.05,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        early_stopping=False,
        random_state=RANDOM_STATE,
    )
    model.fit(X_development, y_development)

    raw_predictions = model.predict(X_test)
    predictions = np.clip(raw_predictions, a_min=0, a_max=None)

    return (
        X_test,
        y_test,
        y_development,
        raw_predictions,
        predictions,
    )


def build_error_results(X_test, y_test, predictions):
    """Собирает таблицу ошибок с диагностическими признаками."""
    columns = [
        "Date",
        "Hour",
        "Seasons",
        "Holiday",
        "Temperature(°C)",
        "Rainfall(mm)",
        "Snowfall (cm)",
    ]

    results = operating_df.loc[X_test.index, columns].copy()
    results["actual"] = y_test.to_numpy()
    results["prediction"] = predictions
    results["residual"] = results["actual"] - results["prediction"]
    results["abs_error"] = results["residual"].abs()
    results["squared_error"] = results["residual"] ** 2

    results["Day type"] = np.where(
        results["Date"].dt.dayofweek >= 5,
        "Weekend",
        "Weekday",
    )

    results["Holiday status"] = np.where(
        results["Holiday"].eq("Holiday"),
        "Holiday",
        "Regular",
    )

    rain = results["Rainfall(mm)"].gt(0)
    snow = results["Snowfall (cm)"].gt(0)

    results["Weather"] = np.select(
        [
            rain & snow,
            snow,
            rain,
        ],
        [
            "Rain + snow",
            "Snow",
            "Rain",
        ],
        default="Dry",
    )

    results["Temperature band"] = pd.cut(
        results["Temperature(°C)"],
        bins=[-np.inf, 0, 10, 20, 30, np.inf],
        labels=[
            "< 0 C",
            "0-10 C",
            "10-20 C",
            "20-30 C",
            ">= 30 C",
        ],
        include_lowest=True,
    )

    results["Demand band"] = pd.cut(
        results["actual"],
        bins=[-np.inf, 300, 700, 1200, np.inf],
        labels=[
            "< 300",
            "300-700",
            "700-1200",
            "1200+",
        ],
        include_lowest=True,
    )

    return results


def summarize_errors(data, group_column):
    """Считает RMSE, MAE и bias по группе наблюдений."""
    summary = data.groupby(
        group_column,
        observed=True,
    ).agg(
        Observations=("residual", "size"),
        MeanSquaredError=("squared_error", "mean"),
        MAE=("abs_error", "mean"),
        Bias=("residual", "mean"),
    )

    summary["RMSE"] = np.sqrt(summary["MeanSquaredError"])

    summary = summary[
        ["Observations", "RMSE", "MAE", "Bias"]
    ]

    summary["Reliable"] = (
        summary["Observations"] >= MIN_GROUP_SIZE
    )

    return summary


def build_segment_table(results):
    """Собирает единую таблицу ошибок по всем разрезам."""
    tables = []

    for segment, column in SEGMENT_GROUPS:
        summary = summarize_errors(
            results,
            column,
        ).reset_index()

        summary = summary.rename(
            columns={column: "Group"}
        )

        summary.insert(0, "Segment", segment)
        tables.append(summary)

    return pd.concat(tables, ignore_index=True)


def build_calibration(results, q=10):
    """Сравнивает средний прогноз и факт по децилям прогноза."""
    calibrated = results.copy()

    calibrated["Prediction band"] = pd.qcut(
        calibrated["prediction"],
        q=q,
        duplicates="drop",
    )

    calibration = calibrated.groupby(
        "Prediction band",
        observed=True,
    ).agg(
        Observations=("residual", "size"),
        MeanPrediction=("prediction", "mean"),
        MeanActual=("actual", "mean"),
        Bias=("residual", "mean"),
        MeanSquaredError=("squared_error", "mean"),
    ).reset_index(drop=True)

    calibration.insert(
        0,
        "Decile",
        np.arange(1, len(calibration) + 1),
    )

    calibration["RMSE"] = np.sqrt(
        calibration["MeanSquaredError"]
    )

    return calibration.drop(columns="MeanSquaredError")


def build_top_errors(results, n=20):
    """Возвращает крупнейшие абсолютные ошибки."""
    top = results.nlargest(n, "abs_error").copy()
    top["Date"] = top["Date"].dt.strftime("%Y-%m-%d")

    return top[
        [
            "Date",
            "Hour",
            "Seasons",
            "actual",
            "prediction",
            "residual",
            "abs_error",
        ]
    ]


def _markdown_table(frame):
    """Форматирует DataFrame как простую markdown-таблицу."""
    display = frame.copy()

    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(
                lambda value: (
                    ""
                    if pd.isna(value)
                    else f"{value:.2f}"
                )
            )

    header = "| " + " | ".join(display.columns) + " |"
    separator = "| " + " | ".join(
        "---" for _ in display.columns
    ) + " |"
    rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in display.itertuples(index=False, name=None)
    ]

    return "\n".join([header, separator, *rows])


def make_figures(results, calibration):
    """Сохраняет графики для презентации."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")

    heatmap_data = results.pivot_table(
        index="Hour",
        columns="Seasons",
        values="squared_error",
        aggfunc=lambda values: np.sqrt(values.mean()),
    )
    heatmap_data = heatmap_data.reindex(
        columns=["Winter", "Spring", "Summer", "Autumn"]
    )

    heatmap_display = heatmap_data.rename(columns=SEASON_LABELS)

    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(
        heatmap_display,
        annot=True,
        fmt=".0f",
        cmap="YlOrRd",
        linewidths=0.5,
        cbar_kws={"label": "RMSE"},
        ax=ax,
    )
    ax.set(
        title="RMSE по часам и сезонам",
        xlabel="Сезон",
        ylabel="Час",
    )
    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "11_error_heatmap.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(
        calibration["MeanPrediction"],
        calibration["MeanActual"],
        marker="o",
        color="#15668b",
    )
    upper = max(
        calibration["MeanPrediction"].max(),
        calibration["MeanActual"].max(),
    )
    axes[0].plot(
        [0, upper],
        [0, upper],
        linestyle="--",
        color="red",
        label="Идеал",
    )
    axes[0].set(
        title="Калибровка по децилям прогноза",
        xlabel="Средний прогноз",
        ylabel="Средний факт",
    )
    axes[0].legend()

    colors = np.where(
        calibration["Bias"] >= 0,
        "#15668b",
        "#c44e52",
    )
    axes[1].bar(
        calibration["Decile"],
        calibration["Bias"],
        color=colors,
    )
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set(
        title="Смещение по децилям прогноза",
        xlabel="Дециль прогноза",
        ylabel="Средняя ошибка: факт − прогноз",
    )

    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "12_calibration_bias.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

    segments = [
        ("Weather", "погоде", WEATHER_LABELS),
        ("Holiday status", "праздникам", HOLIDAY_LABELS),
        ("Day type", "типу дня", DAY_TYPE_LABELS),
        ("Demand band", "фактическому спросу", {}),
    ]
    orders = {
        "Weather": ["Dry", "Rain", "Snow", "Rain + snow"],
        "Holiday status": ["Regular", "Holiday"],
        "Day type": ["Weekday", "Weekend"],
        "Demand band": ["< 300", "300-700", "700-1200", "1200+"],
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for ax, (column, title, labels) in zip(axes.flat, segments):
        summary = summarize_errors(
            results,
            column,
        ).reset_index()

        summary["Group label"] = (
            summary[column].astype(str).map(labels).fillna(
                summary[column].astype(str)
            )
        )

        order = [
            labels.get(group, group)
            for group in orders[column]
        ]
        counts = summary.set_index("Group label")["Observations"]
        reliable = summary.set_index("Group label")["Reliable"]

        sns.barplot(
            data=summary,
            x="Group label",
            y="RMSE",
            order=order,
            color="#15668b",
            ax=ax,
        )

        for patch, group in zip(ax.patches, order):
            count = int(counts.get(group, 0))
            patch.set_facecolor(
                "#15668b" if reliable.get(group, False) else "#9e9e9e"
            )
            ax.annotate(
                f"n={count}",
                (patch.get_x() + patch.get_width() / 2, patch.get_height()),
                ha="center",
                va="bottom",
                fontsize=8,
            )

        ax.set_title(f"RMSE по {title}")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=20)

    fig.text(
        0.5,
        0.01,
        "Серые столбцы: меньше 20 наблюдений",
        ha="center",
        fontsize=9,
    )

    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "13_error_segments.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def write_report(
    overall_metrics,
    segment_table,
    calibration,
    top_errors,
    raw_negative,
):
    """Пишет markdown-отчёт с готовыми формулировками."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    metrics_frame = pd.DataFrame(
        {
            "Metric": list(overall_metrics.keys()),
            "Value": [
                (
                    f"{value:.2f}"
                    if isinstance(value, (float, np.floating))
                    else str(value)
                )
                for value in overall_metrics.values()
            ],
        }
    )

    reliable = segment_table[segment_table["Reliable"]]
    worst_segments = reliable.sort_values(
        "RMSE",
        ascending=False,
    ).head(5)

    non_operating = df.loc[
        df["Functioning Day"] == "No"
    ]

    lines = [
        "# Анализ ошибок финальной модели",
        "",
        "Модель: HistGradientBoosting, random split 60/20/20.",
        "Ошибки посчитаны на test; модель обучена только на operating-часах.",
        "",
        "## Общие метрики",
        "",
        _markdown_table(metrics_frame),
        "",
        f"Отрицательных прогнозов до clipping: {raw_negative}.",
        "",
        "## Где модель ошибается сильнее",
        "",
        _markdown_table(worst_segments),
        "",
        "Сегменты с `Reliable = False` содержат меньше "
        f"{MIN_GROUP_SIZE} наблюдений; по ним не стоит делать выводы.",
        "",
        "## Калибровка по децилям прогноза",
        "",
        _markdown_table(calibration),
        "",
        "Если `MeanActual` ниже `MeanPrediction`, модель завышает спрос; "
        "если выше — занижает.",
        "",
        "## Крупнейшие абсолютные ошибки",
        "",
        _markdown_table(top_errors),
        "",
        "## Нерабочие дни",
        "",
        f"Строк с `Functioning Day = No`: {len(non_operating)}.",
        f"Сумма target: {non_operating['Rented Bike Count'].sum():.0f}.",
        "Эти строки не входят в обучение и не входят в test operating-модели; "
        "для них используется прогноз 0.",
        "",
        "## Что показать в презентации",
        "",
        "1. Тепловую карту ошибок по часу и сезону.",
        "2. Калибровку: где модель завышает и занижает спрос.",
        "3. 2–3 крупнейшие ошибки с контекстом, а не только средний RMSE.",
        "",
    ]

    report_path = REPORTS_DIR / "error_analysis_random.md"
    report_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    segment_table.to_csv(
        REPORTS_DIR / "error_analysis_random.csv",
        index=False,
        encoding="utf-8",
    )

    return report_path


def main():
    (
        X_test,
        y_test,
        y_development,
        raw_predictions,
        predictions,
    ) = fit_final_model()

    results = build_error_results(
        X_test,
        y_test,
        predictions,
    )

    overall_metrics = calculate_metrics(
        y_true=y_test,
        predictions=predictions,
        peak_reference=y_development,
    )

    segment_table = build_segment_table(results)
    calibration = build_calibration(results)
    top_errors = build_top_errors(results)

    make_figures(results, calibration)

    report_path = write_report(
        overall_metrics,
        segment_table,
        calibration,
        top_errors,
        int((raw_predictions < 0).sum()),
    )

    print(f"Report: {report_path}")
    print(
        f"Figures: {FIGURES_DIR / '11_error_heatmap.png'}, "
        f"{FIGURES_DIR / '12_calibration_bias.png'}, "
        f"{FIGURES_DIR / '13_error_segments.png'}"
    )


if __name__ == "__main__":
    main()
