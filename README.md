# MLOps with MLflow

Code and notes from a Udemy MLflow course. The project trains an **ElasticNet regression** model to predict red wine quality and uses **MLflow** to track experiments, parameters, metrics and models.

## Project structure

| File | Purpose |
|---|---|
| `main.py` | Trains the model and logs the run to MLflow |
| `red-wine-quality.csv` | Dataset: 1,599 red wines, 11 physico-chemical features, `quality` target (scores 3–8 in this file), plus an unnamed index column |
| `pyproject.toml` / `uv.lock` | Dependencies, managed with [uv](https://docs.astral.sh/uv/) |
| `.python-version` | Python version (3.12) |
| `requirements.txt` | Plain dependency list for installs without uv |

Not tracked in git (see `.gitignore`): `.venv/`, `mlruns/`, `mlflow.db`, `myartifacts/`, `mlartifacts/`, `data/`.

## Setup

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
git clone https://github.com/saquibherman/mlops-with-mlflow.git
cd mlops-with-mlflow
uv sync
```

`uv sync` creates `.venv/` and installs the exact versions from `uv.lock` (mlflow, scikit-learn, pandas, numpy).

## Running

```bash
uv run main.py                              # defaults: alpha=0.7, l1_ratio=0.7
uv run main.py --alpha 0.5 --l1_ratio 0.3   # custom hyperparameters
```

| Argument | Default | Meaning |
|---|---|---|
| `--alpha` | `0.7` | Overall regularization strength |
| `--l1_ratio` | `0.7` | Mix of L1 vs L2 penalty (0 = ridge, 1 = lasso) |

## What `main.py` does

1. Loads `red-wine-quality.csv` and splits it 75/25 into train and test sets.
2. Sets the MLflow tracking URI to `""`, which means a local file store in `./mlruns`.
3. Calls `mlflow.set_experiment("experiment_1")`. This creates the experiment on the first run and reuses it after that, then prints its name, ID, artifact location, tags, lifecycle stage and creation time.
4. Inside `mlflow.start_run()`:
   - trains `ElasticNet(alpha, l1_ratio, random_state=42)`
   - evaluates on the test set and prints **RMSE**, **MAE** and **R²**
   - logs the params `alpha`, `l1_ratio` and the metrics `rmse`, `mae`, `r2`
   - logs the trained model with `mlflow.sklearn.log_model(lr, "mymodel")`

### Note on the file store (MLflow 3.x)

MLflow 3.x puts the file-based tracking store (`./mlruns`) in maintenance mode and raises an error when it is used. The course uses `uri=""` (file store), so `main.py` sets this before importing mlflow:

```python
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
```

To use the recommended database backend instead, change the tracking URI to `sqlite:///mlflow.db` and remove that line.

## Viewing results

```bash
uv run mlflow ui --backend-store-uri ./mlruns
```

Open http://127.0.0.1:5000 to compare runs, parameters and metrics, and to browse logged models.

Example output with the default parameters:

```
RMSE: 0.8315
MAE:  0.6649
R2:   0.0204
```

## Changelog

Newest first. The full diffs are in the git history (`git log -p --follow main.py`).

### Use set_experiment and the file store (`uri=""`)
- Replaced `create_experiment` + `get_experiment` with `mlflow.set_experiment("experiment_1")`, which creates or reuses the experiment, so re-running no longer fails.
- Switched the tracking URI from `sqlite:///mlflow.db` back to `""` (local `./mlruns`) to match the course.
- Set `MLFLOW_ALLOW_FILE_STORE=true` before importing mlflow, because MLflow 3.x blocks the file store by default.

### Use the SQLite backend and reuse the experiment if it exists
- `uri=""` raised the MLflow 3.x file-store exception, so the tracking URI was changed to `sqlite:///mlflow.db`.
- `create_experiment` failed on a second run because the experiment name already existed. The script now looks the experiment up with `get_experiment_by_name` and only creates it when missing.

### Create an experiment with tags and a custom artifact location
- Created the experiment `exp_create_exp_artifact` with `mlflow.create_experiment`, tags `{"version": "v1", "priority": "p1"}`, and artifacts stored in `./myartifacts`.
- Printed the experiment's metadata.
- Restored the `start_run` block and all param, metric and model logging.
- Default `alpha` and `l1_ratio` set back to 0.7. Removed the copy of the CSV to `data/`.

### Set the tracking URI and change the defaults
- Added `mlflow.set_tracking_uri(uri="")` and a print of the active URI.
- Default `alpha` and `l1_ratio` changed from 0.7 to 0.5.
- Removed the experiment, run and logging code (temporary, during the lesson).

### Rename the script to `main.py`
- `first_ML_code_with_Mlflow.py` → `main.py` (content unchanged).

### Initial commit
- ElasticNet wine-quality model that logs params, metrics and the model to the MLflow experiment `experment_1`.
- Set up as a uv project (`pyproject.toml`, `uv.lock`, `.python-version`) with `.gitignore` for local MLflow data and the virtual environment.
