"""Predictions: load the Champion model from Unity Catalog, predict demand on the test period,
and save the predictions to a Delta table."""

import sys
import logging

import mlflow
import mlflow.catboost
from mlflow import MlflowClient
from sklearn.metrics import r2_score

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
FEATURE_TABLE = "mle_project.gold.supply_chain_features"
PREDICTION_TABLE = "mle_project.gold.supply_chain_predictions"

MODEL_NAME = "mle_project.gold.supply_chain_demand_model"
CHAMPION_ALIAS = "Champion"

TARGET_COLUMN = "total_demand"
DATE_COLUMN = "transaction_date"
KEY_COLUMNS = ["transaction_date", "product_name", "destination_city"]

NUMBER_FEATURE_COLUMNS = ["demand_1", "demand_7", "rolling_7_day_avg", "month", "day_of_week"]
CATEGORY_FEATURE_COLUMNS = ["product_name", "destination_city"]
FEATURE_COLUMNS = NUMBER_FEATURE_COLUMNS + CATEGORY_FEATURE_COLUMNS

TRAIN_END_DATE = "2026-03-23"   # test period = dates after this


# ----------------------------------------------------------------------
# Logger and custom exception
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


logger = get_logger("predictions")


class PredictionError(Exception):
    """Raised when the test data or champion model cannot be loaded, or predictions cannot be saved."""


# ----------------------------------------------------------------------
# Prediction Writer
# ----------------------------------------------------------------------
class PredictionWriter:
    """Predicts demand on the test period with the Champion model and saves the results."""

    def __init__(self, spark, feature_table, prediction_table, model_name, champion_alias, train_end_date, logger):
        self.spark = spark
        self.feature_table = feature_table
        self.prediction_table = prediction_table
        self.model_name = model_name
        self.champion_alias = champion_alias
        self.train_end_date = train_end_date
        self.logger = logger

    def load_test_data(self):
        """Read the test-period rows (after the train end date) into pandas."""
        try:
            self.logger.info("Reading test data from %s", self.feature_table)
            end_date = F.to_date(F.lit(self.train_end_date))
            test_df = (
                self.spark.table(self.feature_table)
                .filter(F.col(DATE_COLUMN) > end_date)
                .select(*KEY_COLUMNS, TARGET_COLUMN, *NUMBER_FEATURE_COLUMNS)
            )
            test_pdf = test_df.toPandas()
            self.logger.info("Test rows loaded: %d", len(test_pdf))
            return test_pdf
        except Exception as e:
            self.logger.error("Failed to load test data: %s", e)
            raise PredictionError(f"Loading test data failed: {e}") from e

    def load_champion(self):
        """Load the model that has the Champion alias, and return it with its version."""
        try:
            client = MlflowClient()
            version = client.get_model_version_by_alias(self.model_name, self.champion_alias).version
            model = mlflow.catboost.load_model(f"models:/{self.model_name}/{version}")
            self.logger.info("Loaded %s version %s", self.champion_alias, version)
            return model, str(version)
        except Exception as e:
            self.logger.error("Failed to load champion model: %s", e)
            raise PredictionError(f"Loading champion model failed: {e}") from e

    def predict(self, model, test_pdf):
        """Predict demand and build the output table."""
        try:
            predicted = model.predict(test_pdf[FEATURE_COLUMNS])
            output_pdf = test_pdf[KEY_COLUMNS].copy()
            output_pdf["actual_demand"] = test_pdf[TARGET_COLUMN].values
            output_pdf["predicted_demand"] = predicted
            self.logger.info("Predictions made for %d rows", len(output_pdf))
            return output_pdf
        except Exception as e:
            self.logger.error("Failed to predict: %s", e)
            raise PredictionError(f"Prediction failed: {e}") from e

    def write_predictions(self, output_pdf, model_version):
        """Save the predictions, with the model version and a timestamp, to the Delta table."""
        try:
            output_df = (
                self.spark.createDataFrame(output_pdf)
                .withColumn("model_version", F.lit(model_version))
                .withColumn("predicted_at", F.current_timestamp())
            )
            self.logger.info("Writing predictions to %s", self.prediction_table)
            (
                output_df.write.format("delta")
                .mode("overwrite")
                .option("overwriteSchema", "true")
                .saveAsTable(self.prediction_table)
            )
            self.logger.info("Write completed: %s", self.prediction_table)
        except Exception as e:
            self.logger.error("Failed to write predictions: %s", e)
            raise PredictionError(f"Writing {self.prediction_table} failed: {e}") from e

    def validate(self, expected_rows):
        """Check the saved table: row count, null predictions and unique keys."""
        saved = self.spark.table(self.prediction_table)
        row_count = saved.count()
        null_predictions = saved.filter(F.col("predicted_demand").isNull()).count()
        distinct_keys = saved.select(*KEY_COLUMNS).distinct().count()
        self.logger.info(
            "Prediction check -> rows: %d | null predictions: %d | distinct keys: %d",
            row_count, null_predictions, distinct_keys,
        )
        if row_count != expected_rows:
            raise PredictionError(f"Row count mismatch: expected={expected_rows}, table={row_count}")
        if null_predictions > 0:
            raise PredictionError(f"{null_predictions} rows have a null prediction")
        if distinct_keys != row_count:
            raise PredictionError("Duplicate date/product/city combinations found in predictions")
        return row_count

    def log_r2_from_saved_table(self):
        """Recalculate R2 from the saved predictions table and log it."""
        check_pdf = (
            self.spark.table(self.prediction_table)
            .select("actual_demand", "predicted_demand")
            .toPandas()
        )
        r2 = r2_score(check_pdf["actual_demand"], check_pdf["predicted_demand"])
        self.logger.info("R2 from the saved table: %.6f", r2)

    def run(self):
        """Run the steps: load test data -> load champion -> predict -> write -> validate -> log R2."""
        try:
            self.logger.info("Starting prediction step")
            test_pdf = self.load_test_data()
            model, model_version = self.load_champion()
            output_pdf = self.predict(model, test_pdf)
            self.write_predictions(output_pdf, model_version)
            self.validate(len(output_pdf))
            self.log_r2_from_saved_table()
            self.logger.info("Predictions saved successfully")
        except PredictionError:
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error in prediction step: {e}")
            raise PredictionError(f"Prediction step failed: {e}") from e


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
def main():
    """Create the Spark session, write the Champion's predictions, and fail loudly on error."""
    spark = SparkSession.builder.getOrCreate()
    mlflow.set_registry_uri("databricks-uc")
    try:
        writer = PredictionWriter(
            spark=spark,
            feature_table=FEATURE_TABLE,
            prediction_table=PREDICTION_TABLE,
            model_name=MODEL_NAME,
            champion_alias=CHAMPION_ALIAS,
            train_end_date=TRAIN_END_DATE,
            logger=logger,
        )
        writer.run()
    except PredictionError as e:
        logger.error(f"Prediction step failed: {e}")
        raise


if __name__ == "__main__":
    main()