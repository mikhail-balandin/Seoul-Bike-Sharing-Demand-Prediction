import argparse
import hashlib
import io
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ..data import TARGET, TIMESTAMP, load_hourly_data
from ..data.features import FEATURE_SETS, build_features
from ..data.protocol import (
    MEMBERSHIP_FILE, ROLLING, TEST_METRICS_FILE, TEST_PREDICTIONS_FILE,
    VALIDATION_MODELS_FILE, make_splits, split_membership, validation_file,
)
from ..paths import RESULTS_DIR
from ..validation.models import make_model
from ..validation.train import baseline_predict, candidate_models, score_predictions, write_scores


def scored_row(
    scheme: str, row: pd.Series, actual: np.ndarray, forecast: np.ndarray, train_hours: int,
    *, winner: bool,
) -> dict:
    scores = score_predictions(actual, forecast)
    scores.pop("validation_hours")
    return {
        "scheme": scheme,
        "feature_set": row["feature_set"],
        "algorithm": row["algorithm"],
        "configuration": row["configuration"],
        "parameters": row["parameters"],
        "train_hours": train_hours,
        "validation_hours": int(row["validation_hours"]),
        "validation_rmse": float(row["rmse"]),
        "test_hours": len(actual),
        **{f"test_{metric}": value for metric, value in scores.items()},
        "is_winner": winner,
    }


def evaluate_winners(results_dir: Path) -> pd.DataFrame:
    splits = make_splits(load_hourly_data())
    expected = split_membership(splits).to_csv(index=False, lineterminator="\n")
    if (results_dir / MEMBERSHIP_FILE).read_text(encoding="utf-8") != expected:
        raise ValueError("Saved split membership differs from the current data. Run train again.")

    models_dir = results_dir / "models"
    retained = joblib.load(models_dir / VALIDATION_MODELS_FILE)
    candidates = candidate_models()
    validation = {scheme: pd.read_csv(results_dir / validation_file(scheme)) for scheme in splits}
    metrics, predictions, saved_models = [], [], []
    for scheme, split in splits.items():
        test = split.test[[TIMESTAMP, TARGET]].rename(columns={TARGET: "actual"}).copy()
        test.insert(0, "scheme", scheme)
        actual = test["actual"].to_numpy()
        scores = validation[scheme]
        winners = scores.drop_duplicates("feature_set").set_index("feature_set", drop=False)

        baseline = baseline_predict(split.final_train, split.test)
        test["baseline"] = baseline
        metrics.append(scored_row(
            scheme, winners.loc["baseline"], actual, baseline, len(split.final_train), winner=False
        ))

        for feature_set, weather in FEATURE_SETS.items():
            winner = winners.loc[feature_set]
            configuration = winner["configuration"]
            if scheme == ROLLING:
                model = make_model(candidates[configuration], weather)
                model.fit(build_features(split.final_train, weather), split.final_train[TARGET])
            else:
                retained_config, model = retained[scheme][feature_set]
                if retained_config != configuration:
                    raise ValueError(f"Retained model does not match {scheme}/{feature_set}. Run train again.")
                check = split.folds["holdout"][1]
                validation_prediction = model.predict(build_features(check, weather)).clip(min=0)
                validation_score = score_predictions(check[TARGET].to_numpy(), validation_prediction)
                if not np.isclose(validation_score["rmse"], winner["rmse"], rtol=0, atol=1e-7):
                    raise ValueError(f"Retained model does not reproduce {scheme}/{feature_set}. Run train again.")

            forecast = model.predict(build_features(split.test, weather)).clip(min=0)
            test[feature_set] = forecast
            row = scored_row(scheme, winner, actual, forecast, len(split.final_train), winner=True)
            metrics.append(row)
            with io.BytesIO() as buffer:
                joblib.dump(model, buffer, compress=3)
                contents = buffer.getvalue()
            checksum = hashlib.sha256(contents).hexdigest()[:12]
            model_file = f"{scheme}_{feature_set}_{checksum}.joblib"
            saved_models.append(
                (model_file, contents, {
                    **row,
                    "parameters": json.loads(winner["parameters"]),
                    "model_file": model_file,
                    "weather_columns": list(weather),
                })
            )

        predictions.append(test)

    for model_file, contents, _ in saved_models:
        path = models_dir / model_file
        if not path.is_file():
            path.write_bytes(contents)
    results = pd.DataFrame(metrics)
    write_scores(results, results_dir / TEST_METRICS_FILE)
    write_scores(pd.concat(predictions, ignore_index=True), results_dir / TEST_PREDICTIONS_FILE)
    manifest = {
        "selection": "validation_rmse_per_scheme",
        "models": [entry for _, _, entry in saved_models],
    }
    manifest_file = models_dir / "manifest.json"
    pending = manifest_file.with_suffix(".tmp")
    pending.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pending.replace(manifest_file)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate each validation winner on its own test split.")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()
    scores = evaluate_winners(args.results_dir)
    columns = ["scheme", "feature_set", "configuration", "validation_rmse", "test_rmse", "test_peak_rmse"]
    print(scores.loc[scores["is_winner"], columns].to_string(index=False))


if __name__ == "__main__":
    main()
