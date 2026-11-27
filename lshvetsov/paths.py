from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR.parent / "SeoulBikeData.csv"
RESULTS_DIR = PROJECT_DIR / "results"
MODEL_DIR = RESULTS_DIR / "models"
