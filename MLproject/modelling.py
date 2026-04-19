import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
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
import datetime

timelapse = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def main():
        # autolog() akan mencatat semua parameter & metrik ke run yang dikelola mlflow run CLI
        mlflow.sklearn.autolog()

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

        # Tidak perlu start_run() — mlflow run CLI sudah mengaktifkan run via env vars
        # Initialize RandomForestRegressor
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=None,
            random_state=42,
            n_jobs=-1
        )
        print("[INFO] Training model: RandomForestRegressor on CPU")

        rf_model.fit(X_train, y_train)

        # Log model ke run yang aktif (dikelola CLI)
        input_example = X_train.iloc[0:5]
        mlflow.sklearn.log_model(
            sk_model=rf_model,
            artifact_path="model",
            input_example=input_example,
        )


if __name__ == "__main__":
    main()