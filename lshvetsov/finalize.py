import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from .data import SELECTION_SCHEMES, TARGET, TEST_START, TIMESTAMP, load_hourly_data
from .features import FEATURE_SETS, build_features, make_model
from .train import (
    RESULTS_DIR,
    baseline_predict,
    candidate_models,
    model_parameters,
    score_predictions,
    write_scores,
)


TEST_END = TEST_START + pd.offsets.MonthBegin()


def select_nominees(results_dir: Path) -> dict[str, dict[str, list[str]]]:
    nominees = {feature_set: {} for feature_set in FEATURE_SETS}
    for scheme in SELECTION_SCHEMES:
        scores = pd.read_csv(results_dir / f"{scheme}.csv")
        winners = (
            scores.loc[scores["feature_set"].isin(FEATURE_SETS)]
            .sort_values(["rmse", "configuration"])
            .drop_duplicates(["feature_set", "algorithm"])
        )
        for row in winners.itertuples(index=False):
            nominees[row.feature_set].setdefault(row.configuration, []).append(scheme)
    return nominees


def evaluate_winners(results_dir: Path) -> pd.DataFrame:
    nominees = select_nominees(results_dir)
    candidates = candidate_models()
    hourly = load_hourly_data()
    is_open = hourly["Functioning Day"].eq("Yes")
    train = hourly.loc[is_open & hourly[TIMESTAMP].lt(TEST_START)]
    test = hourly.loc[hourly[TIMESTAMP].between(TEST_START, TEST_END, inclusive="left")]
    if train.empty or len(test) != int((TEST_END - TEST_START) / pd.Timedelta(hours=1)):
        raise ValueError("The training period and all November test hours are required.")
    open_test = test.loc[is_open.loc[test.index]]
    actual = open_test[TARGET].to_numpy()
    predictions = test[[TIMESTAMP, TARGET]].rename(columns={TARGET: "actual"}).copy()
    predictions["is_open"] = is_open.loc[test.index]

    trial_rows = []
    winners = []
    manifest_models = []
    model_dir = results_dir / "models"
    model_dir.mkdir(exist_ok=True)
    for feature_set, configurations in nominees.items():
        weather = FEATURE_SETS[feature_set]
        train_features = build_features(train, weather)
        test_features = build_features(open_test, weather)
        evaluated = []
        for configuration, schemes in configurations.items():
            estimator = candidates[configuration]
            model = make_model(estimator, weather)
            model.fit(train_features, train[TARGET])
            predicted = model.predict(test_features).clip(min=0)
            row = {
                "feature_set": feature_set,
                "algorithm": type(estimator).__name__,
                "configuration": configuration,
                "parameters": json.dumps(model_parameters(estimator)),
                "selection_schemes": "|".join(schemes),
                **score_predictions(actual, predicted),
                "is_winner": False,
            }
            trial_rows.append(row)
            evaluated.append((row, model, predicted))

        winner, model, predicted = min(
            evaluated, key=lambda trial: (trial[0]["rmse"], trial[0]["configuration"])
        )
        winner["is_winner"] = True
        winners.append(winner)
        predictions[feature_set] = 0.0
        predictions.loc[open_test.index, feature_set] = predicted
        model_file = f"{feature_set}.joblib"
        joblib.dump(model, model_dir / model_file, compress=3)
        parameters = json.loads(winner["parameters"])
        manifest_models.append(
            {
                "feature_set": feature_set,
                "algorithm": winner["algorithm"],
                "alpha": parameters.get("alpha"),
                "parameters": parameters,
                "selection_schemes": winner["selection_schemes"].split("|"),
                "november_rmse": winner["rmse"],
                "november_peak_rmse": winner["peak_rmse"],
                "model_file": model_file,
                "weather_columns": list(weather),
            }
        )

    predicted = baseline_predict(train, open_test)
    predictions["baseline"] = 0.0
    predictions.loc[open_test.index, "baseline"] = predicted
    baseline = {
        "feature_set": "baseline",
        "algorithm": "HourlyMedian",
        "configuration": "baseline",
        "parameters": "{}",
        "selection_schemes": "",
        **score_predictions(actual, predicted),
        "is_winner": False,
    }
    trial_rows.append(baseline)
    winners.append(baseline)
    trial_table = pd.DataFrame(trial_rows).sort_values(["rmse", "configuration"], ignore_index=True)
    family_table = trial_table.drop_duplicates(["feature_set", "algorithm"])
    table = pd.DataFrame(winners).rename(
        columns={
            "rmse": "november_rmse", "mae": "november_mae", "r2": "november_r2",
            "peak_rmse": "november_peak_rmse", "validation_hours": "test_open_hours",
        }
    ).assign(train_hours=len(train))
    write_scores(trial_table, results_dir / "november_candidates.csv")
    write_scores(family_table, results_dir / "november_family_metrics.csv")
    write_scores(table, results_dir / "november_metrics.csv")
    write_scores(predictions, results_dir / "november_predictions.csv")
    manifest = {
        "selection": "november_rmse_across_schemes",
        "train_start": train[TIMESTAMP].min().isoformat(),
        "test_start": TEST_START.isoformat(),
        "test_end_exclusive": TEST_END.isoformat(),
        "models": manifest_models,
    }
    (model_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare nominees from all schemes on November.")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()
    table = evaluate_winners(args.results_dir)
    print(table[["feature_set", "configuration", "selection_schemes", "november_rmse"]].to_string(index=False))


if __name__ == "__main__":
    main()
