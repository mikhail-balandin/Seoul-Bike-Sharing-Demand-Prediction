import argparse
import json
from html import escape
from pathlib import Path
from string import Template

import pandas as pd
import plotly.graph_objects as go

from ..data import RANDOM_SPLIT_SEED, SEOUL_TIMEZONE, TIMESTAMP
from ..data.features import FEATURE_LABELS, FEATURE_SETS
from ..data.protocol import (
    MEMBERSHIP_FILE, MONTHLY, RANDOM, ROLLING, SCHEME_LABELS,
    TEST_METRICS_FILE, TEST_PREDICTIONS_FILE, validation_file,
)
from ..experiments.peak_weights import RESULT_FILE as WEIGHT_RESULT_FILE
from ..paths import RESULTS_DIR
from ..validation.train import (
    BOOSTING_FIXED, BOOSTING_GRID, LASSO_ALPHAS, LASSO_MAX_ITER, PEAK_QUANTILE, RIDGE_ALPHAS,
)


TEMPLATE_PATH = Path(__file__).with_name("report_template.html")
ALGORITHM_LABELS = {
    "HourlyMedian": "Медиана по часу и типу дня",
    "MedianResidualBoosting": "Медиана + поправка бустинга",
}
SCHEME_DESCRIPTIONS = {
    ROLLING: "Прогноз будущих часов по предшествующей истории.",
    MONTHLY: "Точность на случайно отложенных часах с участием всех месяцев.",
    RANDOM: "Точность на случайно отложенных часах знакомого года.",
}
FEATURE_PALETTE = {
    "calendar": "#16837b",
    "calendar_weather_basic": "#a65127",
    "calendar_weather_all": "#5353aa",
    "baseline": "#666d7c",
}
METRIC_LABELS = {"rmse": "RMSE", "mae": "MAE", "r2": "R²", "peak_rmse": "Peak RMSE"}
FOLD_COLUMNS = {
    "rmse_2018_07": "Июль",
    "rmse_2018_08": "Август",
    "rmse_2018_09": "Сентябрь",
    "rmse_2018_10": "Октябрь",
}


def format_decimal(value: float) -> str:
    return f"{value:.2f}".replace(".", ",")


def format_setting(value: float) -> str:
    return format(value, "g").replace(".", ",")


def format_parameters(raw: str) -> str:
    params = json.loads(raw)
    if "max_leaf_nodes" in params:
        return (
            f"{params['max_iter']} итераций · до {params['max_leaf_nodes']} листьев · "
            f"скорость {format_setting(params['learning_rate'])} · "
            f"от {params['min_samples_leaf']} наблюдений в листе"
        )
    if "alpha" in params:
        label = f"α = {format_setting(params['alpha'])} · со свободным членом"
        if "max_iter" in params:
            iterations = f"{params['max_iter']:,}".replace(",", " ")
            return f"{label} · до {iterations} итераций"
        return label
    return "Со свободным членом" if params.get("fit_intercept") else "—"


def render_weighting_results(path: Path) -> str:
    if not path.is_file():
        return "<p>Для таблиц эксперимента запустите модуль <code>lshvetsov.experiments.peak_weights</code>.</p>"

    scores = pd.read_csv(path)
    summary = []
    monthly = []
    months = {"2018_07": "Июль", "2018_08": "Август", "2018_09": "Сентябрь", "2018_10": "Октябрь"}
    for (weighting, setting), group in scores.groupby(["weighting", "setting"], sort=False):
        if weighting == "uniform":
            label = "Текущий бустинг"
        elif weighting == "recency":
            label = f"Давность · вес уменьшается вдвое за {setting} дн."
        else:
            label = f"Пиковые часы · вес ×{setting}"
        periods = group.set_index("period")
        rolling = periods.loc["rolling_cv"]
        november = periods.loc["november"]
        summary.append({
            "Вес обучающих часов": label,
            "RMSE на фолдах": rolling["rmse"],
            "Peak RMSE на фолдах": rolling["peak_rmse"],
            "RMSE в ноябре": november["rmse"],
            "Peak RMSE в ноябре": november["peak_rmse"],
        })
        monthly.extend({
            "Вес обучающих часов": label,
            "Месяц": month,
            "RMSE": periods.at[period, "rmse"],
            "Peak RMSE": periods.at[period, "peak_rmse"],
        } for period, month in months.items())

    summary = pd.DataFrame(summary)
    highlights = []
    for column in ("Peak RMSE на фолдах", "RMSE в ноябре", "Peak RMSE в ноябре"):
        best = summary.loc[summary[column].idxmin()]
        highlights.append(
            f"<li>Минимальная {column} — {format_decimal(best[column])}. "
            f"Вариант «{escape(best['Вес обучающих часов'])}».</li>"
        )
    summary_table = summary.to_html(index=False, border=0, float_format=format_decimal)
    monthly_table = pd.DataFrame(monthly).to_html(index=False, border=0, float_format=format_decimal)
    return (
        f'<ul>{"".join(highlights)}</ul>'
        '<div class="table-scroll" tabindex="0" aria-label="Сравнение весов на фолдах и в ноябре">'
        f'{summary_table}</div>'
        '<details><summary>Метрики по каждому месяцу временных фолдов</summary>'
        '<div class="table-scroll" tabindex="0" aria-label="Сравнение весов по месяцам">'
        f'{monthly_table}</div></details>'
    )


def render_attempts(attempts: pd.DataFrame) -> str:
    rows = attempts.loc[attempts["feature_set"].isin(FEATURE_SETS)]
    winners = rows.drop_duplicates("feature_set").index
    shown = attempts.assign(
        feature_set=attempts["feature_set"].map(FEATURE_LABELS),
        algorithm=attempts["algorithm"].replace(ALGORITHM_LABELS),
        parameters=attempts["parameters"].map(format_parameters),
        result="",
    )
    shown.loc[winners, "result"] = "Победитель"
    columns = {
        "feature_set": "Сценарий",
        "algorithm": "Модель",
        "parameters": "Параметры",
        **METRIC_LABELS,
        "validation_hours": "Проверено часов",
    }
    columns.update({key: month for key, month in FOLD_COLUMNS.items() if key in shown})
    columns["result"] = "Выбор по RMSE"
    return shown[list(columns)].rename(columns=columns).to_html(
        index=False, border=0, float_format=format_decimal
    )


def render_winners(metrics: pd.DataFrame) -> str:
    winners = metrics.loc[metrics["is_winner"]].copy()
    winners["scheme"] = winners["scheme"].map(SCHEME_LABELS)
    winners["feature_set"] = winners["feature_set"].map(FEATURE_LABELS)
    winners["algorithm"] = winners["algorithm"].replace(ALGORITHM_LABELS)
    winners["parameters"] = winners["parameters"].map(format_parameters)
    columns = {
        "scheme": "Схема проверки",
        "feature_set": "Сценарий",
        "algorithm": "Модель",
        "parameters": "Настройки",
        "validation_rmse": "RMSE валидации",
        "test_rmse": "RMSE теста",
        "test_mae": "MAE теста",
        "test_r2": "R² теста",
        "test_peak_rmse": "Peak RMSE теста",
        "test_hours": "Часов в тесте",
    }
    return winners[list(columns)].rename(columns=columns).to_html(
        index=False, border=0, float_format=format_decimal
    )


def open_hours_label(hours: pd.Series) -> str:
    first, last = hours.min(), hours.max()
    start = first.strftime("%d.%m.%Y" if first.hour == 0 else "%d.%m.%Y %H:%M")
    end = last.strftime("%d.%m.%Y" if last.hour == 23 else "%d.%m.%Y %H:%M")
    count = f"{len(hours):,}".replace(",", "\u00a0")
    return f"{start} — {end} · {count} ч"


def render_rolling_schedule(membership: pd.DataFrame) -> tuple[str, str, str]:
    history = membership.loc[membership[ROLLING].eq("train")]
    training, validation = [], []
    folds = history.loc[history["rolling_validation_fold"].ne("")]
    for number, (_, fold) in enumerate(folds.groupby("rolling_validation_fold", sort=True), start=1):
        check = fold[TIMESTAMP]
        train = history.loc[history[TIMESTAMP].lt(check.min()), TIMESTAMP]
        label = f"<li><strong>Фолд {number}.</strong> "
        training.append(f"{label}{open_hours_label(train)}</li>")
        validation.append(f"{label}{open_hours_label(check)}</li>")
    final_training = open_hours_label(history[TIMESTAMP])
    test = membership.loc[membership[ROLLING].eq("test"), TIMESTAMP]
    return (
        f'<ul class="folds">{"".join(training)}</ul>'
        f'<div class="folds-final"><strong>После выбора.</strong> {final_training}</div>',
        f'<ul class="folds">{"".join(validation)}</ul>',
        open_hours_label(test),
    )


def render_splits(metrics: pd.DataFrame) -> str:
    reference = metrics.loc[metrics["feature_set"].eq("baseline")].set_index("scheme")
    rows = []
    for scheme, label in SCHEME_LABELS.items():
        row = reference.loc[scheme]
        rows.append({
            "Схема": label,
            "Что проверяет": SCHEME_DESCRIPTIONS[scheme],
            "Финальное обучение, часов": int(row["train_hours"]),
            "Валидация, часов": int(row["validation_hours"]),
            "Тест, часов": int(row["test_hours"]),
        })
    return pd.DataFrame(rows).to_html(index=False, border=0)


def render_monthly_counts(membership: pd.DataFrame) -> str:
    month = membership[TIMESTAMP].dt.strftime("%Y-%m")
    rows = []
    for scheme, label in SCHEME_LABELS.items():
        counts = pd.crosstab(month, membership[scheme]).reindex(
            columns=["train", "validation", "test"], fill_value=0
        )
        if scheme == ROLLING:
            counts["validation"] = membership["rolling_validation_fold"].ne("").groupby(month).sum()
        counts.insert(0, "Месяц", counts.index)
        counts.insert(0, "Схема", label)
        rows.append(counts)
    labels = {"train": "Финальное обучение", "validation": "Валидация", "test": "Тест"}
    return pd.concat(rows).rename(columns=labels).to_html(index=False, border=0)


def render_peak_segments(predictions: pd.DataFrame) -> str:
    rows = []
    for scheme, hours in predictions.groupby("scheme", sort=False):
        peaks = hours.loc[hours["actual"].ge(hours["actual"].quantile(PEAK_QUANTILE))]
        hour = peaks[TIMESTAMP].dt.hour
        for scenario in FEATURE_SETS:
            squared_error = peaks[scenario].sub(peaks["actual"]).pow(2)
            totals = squared_error.groupby(hour).sum()
            worst_hour = totals.idxmax()
            segment = peaks.loc[hour.eq(worst_hour)]
            share = totals.at[worst_hour] / totals.sum() if totals.sum() else 0
            rows.append({
                "Схема": SCHEME_LABELS[scheme],
                "Сценарий": FEATURE_LABELS[scenario],
                "Час по Сеулу": f"{worst_hour:02d}:00",
                "Пиковых наблюдений": len(segment),
                "Средний факт": segment["actual"].mean(),
                "Средний прогноз": segment[scenario].mean(),
                "Доля суммы квадратов ошибок, %": share * 100,
            })
    return pd.DataFrame(rows).to_html(index=False, border=0, float_format=format_decimal)


def render_peak_chart(metrics: pd.DataFrame) -> str:
    figure = go.Figure()
    for index, scheme in enumerate(SCHEME_LABELS):
        subset = metrics.loc[metrics["scheme"].eq(scheme)].set_index("feature_set")
        labels = [FEATURE_LABELS[key].removeprefix("Календарь + ") for key in FEATURE_PALETTE]
        for metric, name, opacity in (("test_rmse", "Все часы", 0.45), ("test_peak_rmse", "Пиковые часы", 1)):
            figure.add_bar(
                x=subset.loc[list(FEATURE_PALETTE), metric], y=labels, name=name,
                orientation="h", visible=index == 0, marker={"color": list(FEATURE_PALETTE.values()), "opacity": opacity},
                hovertemplate="%{x:.2f} аренд за час",
            )
    buttons = [
        {"label": label, "method": "update", "args": [
            {"visible": [position // 2 == index for position in range(len(figure.data))]}
        ]}
        for index, label in enumerate(SCHEME_LABELS.values())
    ]
    figure.update_layout(
        template="plotly_white", barmode="group", height=360,
        margin={"l": 70, "r": 20, "t": 56, "b": 55},
        font={"family": "system-ui, sans-serif", "color": "#182835"},
        updatemenus=[{"buttons": buttons, "x": 0, "xanchor": "left", "y": 1.17}],
        legend={"orientation": "h", "y": -0.22},
        xaxis_title="RMSE, аренд за час",
        yaxis={"autorange": "reversed", "automargin": True},
    )
    return figure.to_html(
        full_html=False, include_plotlyjs=True, div_id="test-peak-errors",
        config={"displaylogo": False, "responsive": True},
    )


def render_history(results_dir: Path) -> str:
    metrics_path = results_dir / "november_metrics.csv"
    weights_path = results_dir / WEIGHT_RESULT_FILE
    if not metrics_path.is_file() and not weights_path.is_file():
        return ""
    parts = [
        '<details><summary>Анализ пиков в предыдущем ноябрьском прогоне</summary>',
        '<p>Эти результаты получены по прежней схеме с общим ноябрём. '
        'В ней победителей выбирали по ошибке на ноябрьских часах.</p>',
    ]
    if metrics_path.is_file():
        metrics = pd.read_csv(metrics_path)
        labels = {
            "feature_set": "Сценарий",
            "november_rmse": "RMSE в ноябре",
            "november_peak_rmse": "Peak RMSE в ноябре",
        }
        metrics["feature_set"] = metrics["feature_set"].map(FEATURE_LABELS)
        table = metrics[list(labels)].rename(columns=labels).to_html(
            index=False, border=0, float_format=format_decimal
        )
        parts.extend([
            '<div class="table-scroll" tabindex="0" aria-label="Исторические метрики ноября">',
            table, '</div>',
            '<p>Именно в этом прогоне разница между общей ошибкой и ошибкой при высоком спросе '
            'привела к проверке повышенных весов недавних наблюдений и пиковых часов.</p>',
        ])
    if weights_path.is_file():
        parts.extend([
            '<h4>Как веса изменили ошибку в прежнем прогоне</h4>',
            render_weighting_results(weights_path),
        ])
    parts.append('</details>')
    return "\n".join(parts)


def build_report(results_dir: Path) -> Path:
    metrics = pd.read_csv(results_dir / TEST_METRICS_FILE)
    predictions = pd.read_csv(results_dir / TEST_PREDICTIONS_FILE)
    membership = pd.read_csv(results_dir / MEMBERSHIP_FILE, keep_default_na=False)
    for frame in (predictions, membership):
        frame[TIMESTAMP] = pd.to_datetime(frame[TIMESTAMP], utc=True).dt.tz_convert(SEOUL_TIMEZONE)
    validation_tables = {
        f"{scheme}_table": render_attempts(pd.read_csv(results_dir / validation_file(scheme)))
        for scheme in SCHEME_LABELS
    }
    rolling_training, rolling_validation, rolling_test = render_rolling_schedule(membership)
    template = Template(TEMPLATE_PATH.read_text(encoding="utf-8"))
    html = template.substitute(
        winner_count=int(metrics["is_winner"].sum()),
        random_seed=RANDOM_SPLIT_SEED,
        split_table=render_splits(metrics),
        rolling_training=rolling_training,
        rolling_validation=rolling_validation,
        rolling_test=rolling_test,
        winner_table=render_winners(metrics),
        month_counts=render_monthly_counts(membership),
        peak_chart=render_peak_chart(metrics),
        peak_segments_table=render_peak_segments(predictions),
        history=render_history(results_dir),
        ridge_alphas=" / ".join(map(format_setting, RIDGE_ALPHAS)),
        lasso_alphas=" / ".join(map(format_setting, LASSO_ALPHAS)),
        lasso_max_iter=f"{LASSO_MAX_ITER:,}".replace(",", " "),
        boosting_iterations=" / ".join(map(format_setting, BOOSTING_GRID["max_iter"])),
        boosting_leaves=" / ".join(map(format_setting, BOOSTING_GRID["max_leaf_nodes"])),
        boosting_rates=" / ".join(map(format_setting, BOOSTING_GRID["learning_rate"])),
        boosting_min_leaf=BOOSTING_FIXED["min_samples_leaf"],
        boosting_seed=BOOSTING_FIXED["random_state"],
        **validation_tables,
    )
    path = results_dir / "report.html"
    path.write_text(html, encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the standalone rental report.")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()
    print(build_report(args.results_dir))


if __name__ == "__main__":
    main()
