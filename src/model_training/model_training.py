"""Model training: train CatBoost, log it to MLflow, register it in Unity Catalog,
then compare the new Challenger with the current Champion and promote it if it is better."""

import sys
import logging

import mlflow
import mlflow.catboost
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mlflow.exceptions import MlflowException
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from catboost import CatBoostRegressor

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
FEATURE_TABLE = "mle_project.gold.supply_chain_features"

TARGET_COLUMN = "total_demand"
DATE_COLUMN = "transaction_date"

NUMBER_FEATURE_COLUMNS = ["demand_1", "demand_7", "rolling_7_day_avg", "month", "day_of_week"]
CATEGORY_FEATURE_COLUMNS = ["product_name", "destination_city"]
FEATURE_COLUMNS = NUMBER_FEATURE_COLUMNS + CATEGORY_FEATURE_COLUMNS

TRAIN_END_DATE = "2026-03-23"   # train = up to this date, test = after it

EXPERIMENT_FILE_NAME = "supply_chain_demand_catboost"   # full path is /Users/<current_user>/<this name>

MODEL_PARAMS = {
    "iterations": 500,
    "learning_rate": 0.05,
    "depth": 6,
    "random_seed": 42,
    "verbose": 100,
}

MODEL_NAME = "mle_project.gold.supply_chain_demand_model"
CHAMPION_ALIAS = "Champion"
CHALLENGER_ALIAS = "Challenger"
MIN_IMPROVEMENT = 0.0


# ----------------------------------------------------------------------
# Logger and custom exceptions
# ----------------------------------------------------------------------
def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
        logger.addHandler(handler)
    logger.propagate = False
    return logger


logger = get_logger("model_training")


class ModelTrainingError(Exception):
    """Raised when the model data cannot be loaded, split, trained or logged."""


class ModelPromotionError(Exception):
    """Raised when the challenger cannot be registered, compared or promoted."""


# ----------------------------------------------------------------------
# Data Loader
# ----------------------------------------------------------------------
class DemandDataLoader:
    """Loads the feature table and splits it into train and test data."""

    def __init__(self, spark, feature_table, train_end_date):
        self.spark = spark
        self.feature_table = feature_table
        self.train_end_date = train_end_date

    def load_features(self):
        """Read the feature table and keep only the columns we need."""
        logger.info("Reading feature table %s", self.feature_table)
        return self.spark.table(self.feature_table).select(
            DATE_COLUMN, TARGET_COLUMN, *FEATURE_COLUMNS
        )

    def split_by_date(self, df):
        """Train = up to train_end_date, test = after it."""
        end_date = F.to_date(F.lit(self.train_end_date))
        train_df = df.filter(F.col(DATE_COLUMN) <= end_date)
        test_df = df.filter(F.col(DATE_COLUMN) > end_date)

        train_rows = train_df.count()
        test_rows = test_df.count()
        logger.info("Train rows: %s | Test rows: %s", train_rows, test_rows)

        if train_rows == 0 or test_rows == 0:
            raise ModelTrainingError("Train or test data is empty")
        return train_df, test_df

    def to_pandas(self, train_df, test_df):
        """Drop the date column and convert both parts to pandas."""
        train_pdf = train_df.drop(DATE_COLUMN).toPandas()
        test_pdf = test_df.drop(DATE_COLUMN).toPandas()
        logger.info("Converted to pandas")
        return train_pdf, test_pdf

    def run(self):
        """Run all steps and return train and test pandas DataFrames."""
        try:
            df = self.load_features()
            train_df, test_df = self.split_by_date(df)
            return self.to_pandas(train_df, test_df)
        except ModelTrainingError:
            raise
        except Exception as e:
            logger.error("Data loading failed: %s", e)
            raise ModelTrainingError("Data loading failed") from e


# ----------------------------------------------------------------------
# Model Trainer
# ----------------------------------------------------------------------
class DemandModelTrainer:
    """Trains CatBoost, logs metrics and artifacts to MLflow, registers the model in Unity Catalog."""

    def __init__(self, spark, experiment_name, model_params, model_name):
        self.spark = spark
        self.experiment_name = experiment_name
        self.model_params = model_params
        self.model_name = model_name

    def split_features_target(self, pdf):
        """Separate input columns (X) from the target column (y)."""
        return pdf[FEATURE_COLUMNS], pdf[TARGET_COLUMN]

    def train_model(self, x_train, y_train):
        """Train CatBoost. It handles text columns and nulls by itself."""
        logger.info("Training CatBoost on %s rows", len(x_train))
        model = CatBoostRegressor(
            **self.model_params,
            cat_features=CATEGORY_FEATURE_COLUMNS,
        )
        model.fit(x_train, y_train)
        return model

    def evaluate_model(self, model, x_test, y_test):
        """Predict on the test data and calculate R2, RMSE and MAE."""
        predictions = model.predict(x_test)
        metrics = {
            "r2": r2_score(y_test, predictions),
            "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
            "mae": mean_absolute_error(y_test, predictions),
        }
        logger.info(
            "Test R2: %.4f | RMSE: %.4f | MAE: %.4f",
            metrics["r2"], metrics["rmse"], metrics["mae"],
        )
        return metrics, predictions

    def get_feature_table_version(self):
        """Delta version of the feature table used for this training run."""
        history = self.spark.sql(f"DESCRIBE HISTORY {FEATURE_TABLE} LIMIT 1")
        return history.collect()[0]["version"]

    def log_artifacts(self, model, x_test, y_test, predictions):
        """Log feature importance, actual vs predicted chart and test predictions."""
        importance = pd.DataFrame({
            "feature": model.feature_names_,
            "importance": model.get_feature_importance(),
        }).sort_values("importance", ascending=False)
        mlflow.log_text(importance.to_csv(index=False), "feature_importance.csv")

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(importance["feature"], importance["importance"])
        ax.invert_yaxis()
        ax.set_title("Feature importance")
        mlflow.log_figure(fig, "feature_importance.png")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(y_test, predictions, s=5, alpha=0.4)
        limits = [min(y_test.min(), predictions.min()), max(y_test.max(), predictions.max())]
        ax.plot(limits, limits, color="red")
        ax.set_xlabel("Actual demand")
        ax.set_ylabel("Predicted demand")
        ax.set_title("Actual vs predicted (test set)")
        mlflow.log_figure(fig, "actual_vs_predicted.png")
        plt.close(fig)

        test_predictions = x_test.copy()
        test_predictions["actual_demand"] = y_test.values
        test_predictions["predicted_demand"] = predictions
        mlflow.log_text(test_predictions.to_csv(index=False), "test_predictions.csv")

    def log_and_register(self, model, x_train, x_test, y_test, predictions, metrics):
        """Log to MLflow and register a new model version in Unity Catalog."""
        signature = infer_signature(x_train, model.predict(x_train))
        mlflow.set_experiment(self.experiment_name)
        with mlflow.start_run(run_name="catboost_demand_model"):
            mlflow.log_params(self.model_params)
            mlflow.set_tags({
                "feature_table": FEATURE_TABLE,
                "feature_table_version": str(self.get_feature_table_version()),
                "train_end_date": TRAIN_END_DATE,
                "train_rows": str(len(x_train)),
                "test_rows": str(len(x_test)),
            })
            mlflow.log_metrics(metrics)
            self.log_artifacts(model, x_test, y_test, predictions)
            model_info = mlflow.catboost.log_model(
                model,
                artifact_path="model",
                signature=signature,
                input_example=x_train.head(5),
                registered_model_name=self.model_name,
            )
        version = model_info.registered_model_version
        logger.info("Registered %s version %s", self.model_name, version)
        return version

    def run(self, train_pdf, test_pdf):
        """Run all steps and return the metrics and the new model version."""
        try:
            x_train, y_train = self.split_features_target(train_pdf)
            x_test, y_test = self.split_features_target(test_pdf)
            model = self.train_model(x_train, y_train)
            metrics, predictions = self.evaluate_model(model, x_test, y_test)
            version = self.log_and_register(
                model, x_train, x_test, y_test, predictions, metrics
            )
            return metrics, version
        except ModelTrainingError:
            raise
        except Exception as e:
            logger.error("Model training failed: %s", e)
            raise ModelTrainingError("Model training failed") from e


# ----------------------------------------------------------------------
# Model Promoter (Champion vs Challenger)
# ----------------------------------------------------------------------
class ModelPromoter:
    """Compares a new Challenger model with the current Champion and promotes it if it is better."""

    def __init__(self, model_name, champion_alias, challenger_alias, min_improvement=0.0):
        self.model_name = model_name
        self.champion_alias = champion_alias
        self.challenger_alias = challenger_alias
        self.min_improvement = min_improvement
        self.client = MlflowClient()

    def set_challenger(self, version, r2):
        """Put the Challenger alias on the new version and store its R2 as a tag."""
        self.client.set_registered_model_alias(self.model_name, self.challenger_alias, version)
        self.client.set_model_version_tag(self.model_name, version, "test_r2", str(r2))
        logger.info("Version %s is now the %s (R2 %.4f)", version, self.challenger_alias, r2)

    def get_champion_version(self):
        """Return the version that has the Champion alias, or None if there is no champion yet."""
        try:
            champion = self.client.get_model_version_by_alias(self.model_name, self.champion_alias)
            return str(champion.version)
        except MlflowException as e:
            logger.info("No %s found yet: %s", self.champion_alias, e)
            return None

    def score_champion(self, x_test, y_test):
        """Load the champion and calculate its R2 on the same test data."""
        champion_model = mlflow.catboost.load_model(
            f"models:/{self.model_name}@{self.champion_alias}"
        )
        predictions = champion_model.predict(x_test)
        return r2_score(y_test, predictions)

    def promote(self, version):
        """Move the Champion alias to this version and remove the Challenger alias."""
        self.client.set_registered_model_alias(self.model_name, self.champion_alias, version)
        self.client.delete_registered_model_alias(self.model_name, self.challenger_alias)
        logger.info("Version %s is now the %s", version, self.champion_alias)

    def run(self, challenger_version, challenger_r2, x_test, y_test):
        """Compare the challenger with the champion and promote it if it is better."""
        try:
            challenger_version = str(challenger_version)
            self.set_challenger(challenger_version, challenger_r2)

            champion_version = self.get_champion_version()

            if champion_version is None:
                self.promote(challenger_version)
                return {"decision": "promoted (first model)", "champion_version": challenger_version,
                        "champion_r2": None, "challenger_r2": challenger_r2}

            champion_r2 = self.score_champion(x_test, y_test)
            logger.info(
                "Champion v%s R2: %.4f | Challenger v%s R2: %.4f",
                champion_version, champion_r2, challenger_version, challenger_r2,
            )

            if challenger_r2 > champion_r2 + self.min_improvement:
                self.promote(challenger_version)
                decision, winner = "promoted", challenger_version
            else:
                logger.info("Challenger is not better, champion stays")
                decision, winner = "kept champion", champion_version

            return {"decision": decision, "champion_version": winner,
                    "champion_r2": champion_r2, "challenger_r2": challenger_r2}
        except Exception as e:
            logger.error("Model promotion failed: %s", e)
            raise ModelPromotionError("Model promotion failed") from e


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
def main():
    """Train and register a Challenger, then decide whether it replaces the Champion."""
    spark = SparkSession.builder.getOrCreate()
    mlflow.set_registry_uri("databricks-uc")

    current_user = spark.sql("SELECT current_user()").first()[0]
    experiment_name = f"/Users/{current_user}/{EXPERIMENT_FILE_NAME}"

    try:
        loader = DemandDataLoader(spark, FEATURE_TABLE, TRAIN_END_DATE)
        train_pdf, test_pdf = loader.run()
        logger.info("Train shape: %s | Test shape: %s", train_pdf.shape, test_pdf.shape)

        trainer = DemandModelTrainer(spark, experiment_name, MODEL_PARAMS, MODEL_NAME)
        challenger_metrics, challenger_version = trainer.run(train_pdf, test_pdf)
        logger.info("Challenger metrics: %s", challenger_metrics)
        logger.info("Challenger version: %s", challenger_version)
    except ModelTrainingError as e:
        logger.error("Pipeline stopped: %s", e)
        raise

    try:
        x_test, y_test = trainer.split_features_target(test_pdf)
        promoter = ModelPromoter(MODEL_NAME, CHAMPION_ALIAS, CHALLENGER_ALIAS, MIN_IMPROVEMENT)
        result = promoter.run(challenger_version, challenger_metrics["r2"], x_test, y_test)
        logger.info("Promotion result: %s", result)
    except ModelPromotionError as e:
        logger.error("Pipeline stopped: %s", e)
        raise


if __name__ == "__main__":
    main()