import time
import tracemalloc
import mlflow
import numpy as np
import pandas as pd
import sklearn.utils
# Monkey patch for scikit-learn compatibility with skopt
if not hasattr(sklearn.utils, 'check_pandas_support'):
    try:
        from sklearn.utils import _optional_dependencies
        sklearn.utils.check_pandas_support = _optional_dependencies.check_pandas_support
    except ImportError:
        def check_pandas_support():
            try:
                import pandas
                return pandas
            except ImportError:
                return None
        sklearn.utils.check_pandas_support = check_pandas_support

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import json
import datetime

timelapse = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def safe_mape(y_true, y_pred, epsilon=1e-8):
    """Compute MAPE safely by avoiding division by values near zero."""
    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)
    denominator = np.maximum(np.abs(y_true_array), epsilon)
    return float(np.mean(np.abs((y_true_array - y_pred_array) / denominator)))


def adjusted_r2_score(r2_value, n_samples, n_features):
    """Compute adjusted R2 when degrees of freedom are sufficient."""
    if n_samples <= n_features + 1:
        return float("nan")
    return float(1 - ((1 - r2_value) * (n_samples - 1) / (n_samples - n_features - 1)))


# Removed GPU functions


def build_bayesian_search(estimator):
    from skopt import BayesSearchCV
    from skopt.space import Integer, Real

    search_spaces = {
        "n_estimators": Integer(100, 1000),
        "max_depth": Integer(5, 50),
        "min_samples_split": Integer(2, 20),
        "min_samples_leaf": Integer(1, 10),
        "max_features": Real(0.1, 1.0),
    }

    scoring = {
        "mse": "neg_mean_squared_error",
        "mae": "neg_mean_absolute_error",
        "r2": "r2"
    }

    return BayesSearchCV(
        estimator=estimator,
        search_spaces=search_spaces,
        n_iter=40,
        cv=5,
        n_jobs=-1,
        random_state=42,
        scoring=scoring,
        refit="mse",
        verbose=1,
        return_train_score=True,
    )


def log_cv_iteration_metrics(search_obj):
    """Log every CV candidate result so charts show full train/eval history."""
    cv_results = search_obj.cv_results_
    total_candidates = len(cv_results["mean_test_mse"])
    best_so_far_mse = float("inf")

    # split_test_keys = sorted(
    #     [key for key in cv_results if re.match(r"split\d+_test_score", key)],
    #     key=lambda item: int(re.findall(r"\d+", item)[0]),
    # )
    # split_train_keys = sorted(
    #     [key for key in cv_results if re.match(r"split\d+_train_score", key)],
    #     key=lambda item: int(re.findall(r"\d+", item)[0]),
    # )

    for step_idx in range(total_candidates):
        # mean_val_mse = -float(cv_results["mean_test_score"][step_idx])
        # mlflow.log_metric("cv_mean_val_mse", mean_val_mse, step=step_idx)
        # mlflow.log_metric(
        #     "cv_std_val_score",
        #     float(cv_results["std_test_score"][step_idx]),
        #     step=step_idx,
        # )

        mlflow.log_metric("mse", -cv_results["mean_test_mse"][step_idx], step=step_idx)
        mlflow.log_metric("mae", -cv_results["mean_test_mae"][step_idx], step=step_idx)
        mlflow.log_metric("r2", cv_results["mean_test_r2"][step_idx], step=step_idx)
        rmse = np.sqrt(-cv_results["mean_test_mse"][step_idx])
        mlflow.log_metric("rmse", rmse, step=step_idx)

        mlflow.log_metric("fit_time_sec", cv_results["mean_fit_time"][step_idx], step=step_idx)
        mlflow.log_metric("score_time_sec", cv_results["mean_score_time"][step_idx], step=step_idx)

        # best_so_far_mse = min(best_so_far_mse, mean_val_mse)
        # mlflow.log_metric("cv_best_so_far_mse", best_so_far_mse, step=step_idx)

def main():
        
        run_start_time = time.perf_counter()
        tracemalloc.start()

        # Load data
        df = pd.read_csv("./housing_preprocessing.csv")
        X = df.drop("median_house_value", axis=1)
        y = df["median_house_value"]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
        )

        # Initialize RandomForestRegressor
        rf_model = RandomForestRegressor(random_state=42, n_jobs=-1)
        print("[INFO] Training device: CPU")
        mlflow.log_param("training_device", "CPU")

        bayes_search = build_bayesian_search(rf_model)
        bayes_search.fit(X_train, y_train)
        
        log_cv_iteration_metrics(bayes_search)
        best_model = bayes_search.best_estimator_

        y_pred = best_model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = float(np.sqrt(mse))
        r2 = r2_score(y_test, y_pred)
        mape = safe_mape(y_test, y_pred)
        adjusted_r2 = adjusted_r2_score(
            r2_value=r2,
            n_samples=X_test.shape[0],
            n_features=X_test.shape[1],
        )

        mlflow.log_param("model_name", "RandomForestRegressor")
        mlflow.log_param("workflow", "model_only_no_pipeline")
        mlflow.log_param("tuning_method", "BayesSearchCV")
        mlflow.log_param("scoring", "neg_mean_squared_error")
        mlflow.log_param("cv", 5)
        mlflow.log_param("n_iter", 40)
        mlflow.log_param("random_state", 42)
        mlflow.log_params(bayes_search.best_params_)

        # mlflow.log_metric("test_mae", mae)
        # mlflow.log_metric("test_mse", mse)
        # mlflow.log_metric("test_rmse", rmse)
        # mlflow.log_metric("test_r2", r2)
        # mlflow.log_metric("test_mape", mape)
        # if not np.isnan(adjusted_r2):
        #     mlflow.log_metric("test_adjusted_r2", adjusted_r2)


        mlflow.log_metric("best_cv_mse", -bayes_search.best_score_)
        mlflow.log_metric("best_cv_rmse", float(np.sqrt(-bayes_search.best_score_)))

        with open("best_params.json", "w") as f:
            json.dump(bayes_search.best_params_, f)

        # log_system_metrics(run_start_time, training_device)
        tracemalloc.stop()

        input_example = X_train.iloc[0:5]
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            input_example=input_example,
            # registered_model_name="RandomForestRegressor"
        )

        if rmse < 100000:
            result = mlflow.register_model(
                model_uri=f"runs/{mlflow.active_run().info.run_id}/model",
                name="RandomForestRegressor"
            )

        # result = mlflow.register_model(
        #   model_uri=f""
        # )

        # mlflow.log_artifact(f"models/xgb_tuned_model_{training_device.lower()}_{timelapse}.json")


if __name__ == "__main__":
    # mlflow.set_tracking_uri("http://127.0.0.1:5000")
    # mlflow.set_experiment("Final submission MSML Tunned")
    main()
