import argparse
import json
from html import escape
from pathlib import Path
from string import Template

import pandas as pd
import plotly.graph_objects as go

from .data import RANDOM_SPLIT_SEED, SELECTION_SCHEMES, SEOUL_TIMEZONE, TIMESTAMP
from .features import FEATURE_LABELS, FEATURE_SETS
from .train import (
    BOOSTING_FIXED, BOOSTING_GRID, LASSO_ALPHAS, LASSO_MAX_ITER,
    PEAK_QUANTILE, RESULTS_DIR, RIDGE_ALPHAS,
)


TEMPLATE_PATH = Path(__file__).with_name("report_template.html")
ALGORITHM_LABELS = {
    "HourlyMedian": "Медиана по часу и типу дня",
    "MedianResidualBoosting": "Медиана + поправка бустинга",
}
FAMILY_COLUMNS = {
    "LinearRegression": "Линейная регрессия",
    "Ridge": "Ridge",
    "Lasso": "Lasso",
    "HistGradientBoostingRegressor": "Бустинг напрямую",
    "MedianResidualBoosting": "Медиана + бустинг",
    "HourlyMedian": "Почасовая медиана",
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


def format_schemes(schemes: str) -> str:
    return " · ".join(SELECTION_SCHEMES[scheme] for scheme in schemes.split("|") if scheme)


def render_attempts(
    attempts: pd.DataFrame, *, november: bool = False, show_folds: bool = False
) -> str:
    columns = {
        "feature_set": "Сценарий",
        "algorithm": "Модель",
        "parameters": "Параметры",
        **METRIC_LABELS,
        "validation_hours": "Часов",
    }
    displayed = attempts.assign(
        feature_set=attempts["feature_set"].map(FEATURE_LABELS),
        algorithm=attempts["algorithm"].replace(ALGORITHM_LABELS),
        parameters=attempts["parameters"].map(format_parameters),
    )
    if show_folds:
        columns.update(FOLD_COLUMNS)
    if november:
        displayed["selection_schemes"] = attempts["selection_schemes"].fillna("").map(format_schemes)
        displayed["result"] = attempts["is_winner"].map({True: "Выбрана", False: ""})
        columns.update({"selection_schemes": "Схемы отбора", "result": "Итог"})
    else:
        displayed["result"] = ""
        eligible = attempts.loc[attempts["feature_set"].isin(FEATURE_SETS)]
        best = eligible.drop_duplicates(["feature_set", "algorithm"])
        displayed.loc[best.index, "result"] = "Кандидат"
        columns["result"] = "Итог"
    return displayed[list(columns)].rename(columns=columns).to_html(
        index=False, border=0, na_rep="—", float_format=format_decimal
    )


def render_results(november: pd.DataFrame) -> str:
    cards = []
    for row in november.loc[november["feature_set"].isin(FEATURE_SETS)].to_dict("records"):
        metrics = "".join(
            f"<div><dt>{label}</dt><dd>{format_decimal(row[f'november_{key}'])}</dd></div>"
            for key, label in METRIC_LABELS.items()
        )
        algorithm = ALGORITHM_LABELS.get(row["algorithm"], row["algorithm"])
        cards.append(
            f'<article class="result-card"><h3>{escape(FEATURE_LABELS[row["feature_set"]])}</h3>'
            f'<p><strong>{escape(algorithm)}</strong><br>'
            f'{escape(format_parameters(row["parameters"]))}</p>'
            f'<p class="scheme">Схемы отбора<br>{escape(format_schemes(row["selection_schemes"]))}</p>'
            f'<dl class="metrics">{metrics}</dl></article>'
        )
    return "\n".join(cards)


def render_family_comparison(validation: dict[str, pd.DataFrame], november: pd.DataFrame) -> str:
    rows = []
    schemes = [(SELECTION_SCHEMES[scheme], scores) for scheme, scores in validation.items()]
    for label, scores in [*schemes, ("Весь ноябрь", november)]:
        baseline = scores.loc[scores["algorithm"].eq("HourlyMedian")].iloc[0]
        for feature_set in FEATURE_SETS:
            best = (
                scores.loc[scores["feature_set"].eq(feature_set)]
                .sort_values(["rmse", "configuration"])
                .drop_duplicates("algorithm")
                .set_index("algorithm")
            )
            row = {"Схема": label, "Сценарий": FEATURE_LABELS[feature_set]}
            for algorithm, column in FAMILY_COLUMNS.items():
                score = baseline if algorithm == "HourlyMedian" else best.loc[algorithm]
                row[column] = f"{format_decimal(score['rmse'])} / {format_decimal(score['peak_rmse'])}"
            rows.append(row)
    return pd.DataFrame(rows).to_html(index=False, border=0)


def render_chart(
    figure: go.Figure, name: str, *, include_plotly: bool = False, height: int = 360
) -> str:
    figure.update_layout(
        template="plotly_white",
        height=height,
        margin={"l": 66, "r": 24, "t": 18, "b": 54},
        legend={"orientation": "h", "y": -0.24},
        font={"family": "system-ui, sans-serif", "color": "#182835"},
    )
    return figure.to_html(
        full_html=False, include_plotlyjs=include_plotly, div_id=name,
        config={"displaylogo": False, "responsive": True},
    )


def render_week(predictions: pd.DataFrame) -> str:
    first_monday = pd.Timestamp("2018-11-05", tz=SEOUL_TIMEZONE)
    week = predictions.loc[predictions[TIMESTAMP].between(
        first_monday, first_monday + pd.Timedelta(days=7), inclusive="left"
    )]
    hours = week[TIMESTAMP].dt.tz_localize(None)
    figure = go.Figure()
    for name, color in {"actual": "#182835", **FEATURE_PALETTE}.items():
        figure.add_scatter(
            x=hours,
            y=week[name],
            name=FEATURE_LABELS.get(name, "Факт"),
            mode="lines",
            line={"color": color, "width": 2.6 if name == "actual" else 1.8},
            visible="legendonly" if name == "baseline" else True,
            hovertemplate="%{y:.0f} аренд",
        )
    figure.update_layout(
        hovermode="x unified",
        xaxis={"tickformat": "%d.%m", "hoverformat": "%d.%m %H:%M"},
        yaxis_title="Аренд за час",
    )
    return render_chart(figure, "november-week", include_plotly=True, height=380)


def render_peaks(november: pd.DataFrame) -> str:
    labels = november["feature_set"].map(FEATURE_LABELS).str.removeprefix("Календарь + ")
    colors = november["feature_set"].map(FEATURE_PALETTE).tolist()
    figure = go.Figure()
    for metric, name, opacity in (("rmse", "Все часы", 0.45), ("peak_rmse", "Пиковые часы", 1)):
        figure.add_bar(
            y=labels,
            x=november[f"november_{metric}"],
            name=name,
            orientation="h",
            marker={"color": colors, "opacity": opacity},
            hovertemplate="%{x:.2f} аренд",
        )
    figure.update_layout(
        barmode="group", xaxis_title="RMSE, аренд за час",
        yaxis={"autorange": "reversed", "automargin": True}, bargap=0.35,
    )
    return render_chart(figure, "november-peaks")


def build_report(results_dir: Path) -> Path:
    validation = {
        scheme: pd.read_csv(results_dir / f"{scheme}.csv") for scheme in SELECTION_SCHEMES
    }
    november = pd.read_csv(results_dir / "november_metrics.csv")
    nominees = pd.read_csv(results_dir / "november_candidates.csv")
    family_scores = pd.read_csv(results_dir / "november_family_metrics.csv")
    finalists = (
        family_scores.loc[family_scores["feature_set"].isin(FEATURE_SETS)]
        .groupby("feature_set", sort=False).head(3)
    )
    predictions = pd.read_csv(results_dir / "november_predictions.csv")
    predictions[TIMESTAMP] = pd.to_datetime(
        predictions[TIMESTAMP], utc=True
    ).dt.tz_convert(SEOUL_TIMEZONE)
    open_hours = predictions.loc[predictions["is_open"]]
    peak_threshold = open_hours["actual"].quantile(PEAK_QUANTILE)
    peak_count = int(open_hours["actual"].ge(peak_threshold).sum())
    baseline = november.loc[november["feature_set"].eq("baseline"), "november_rmse"].item()
    best_peak = november.loc[november["november_peak_rmse"].idxmin()]
    full_weather = family_scores.loc[family_scores["feature_set"].eq("calendar_weather_all")]
    full_weather = full_weather.set_index("algorithm")

    template = Template(TEMPLATE_PATH.read_text(encoding="utf-8"))
    html = template.substitute(
        results_cards=render_results(november),
        baseline_rmse=format_decimal(baseline),
        test_hours=len(open_hours),
        train_hours=int(november["train_hours"].iloc[0]),
        rolling_hours=int(validation["rolling_cv"]["validation_hours"].iloc[0]),
        monthly_hours=int(validation["monthly_80_20"]["validation_hours"].iloc[0]),
        random_hours=int(validation["random_80_20"]["validation_hours"].iloc[0]),
        random_seed=RANDOM_SPLIT_SEED,
        ridge_alphas=" / ".join(map(format_setting, RIDGE_ALPHAS)),
        lasso_alphas=" / ".join(map(format_setting, LASSO_ALPHAS)),
        lasso_max_iter=f"{LASSO_MAX_ITER:,}".replace(",", " "),
        configuration_count=len(validation["rolling_cv"]) - 1,
        family_comparison_table=render_family_comparison(validation, family_scores),
        november_direct_rmse=format_decimal(full_weather.at["HistGradientBoostingRegressor", "rmse"]),
        november_direct_peak=format_decimal(full_weather.at["HistGradientBoostingRegressor", "peak_rmse"]),
        november_residual_rmse=format_decimal(full_weather.at["MedianResidualBoosting", "rmse"]),
        november_residual_peak=format_decimal(full_weather.at["MedianResidualBoosting", "peak_rmse"]),
        boosting_iterations=" / ".join(map(format_setting, BOOSTING_GRID["max_iter"])),
        boosting_leaves=" / ".join(map(format_setting, BOOSTING_GRID["max_leaf_nodes"])),
        boosting_rates=" / ".join(map(format_setting, BOOSTING_GRID["learning_rate"])),
        boosting_min_leaf=BOOSTING_FIXED["min_samples_leaf"],
        boosting_seed=BOOSTING_FIXED["random_state"],
        rolling_table=render_attempts(validation["rolling_cv"], show_folds=True),
        monthly_table=render_attempts(validation["monthly_80_20"]),
        random_table=render_attempts(validation["random_80_20"]),
        november_table=render_attempts(finalists, november=True),
        november_attempts_table=render_attempts(nominees, november=True),
        week_chart=render_week(predictions),
        peaks_chart=render_peaks(november),
        peak_threshold=format_decimal(peak_threshold),
        peak_count=peak_count,
        best_peak_model=FEATURE_LABELS[best_peak["feature_set"]],
        best_peak_rmse=format_decimal(best_peak["november_peak_rmse"]),
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
