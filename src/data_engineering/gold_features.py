"""Gold layer: build the daily aggregate table, then the feature table used for model training."""

import sys
import logging

from pyspark.sql import SparkSession
from pyspark.sql import Window
from pyspark.sql import functions as F

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
SOURCE_TABLE  = "mle_project.silver.supply_chain_clean"
AGG_TABLE     = "mle_project.gold.supply_chain_daily_agg"
FEATURE_TABLE = "mle_project.gold.supply_chain_features"


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


logger = get_logger("gold_features")


class GoldError(Exception):
    """Raised when the Gold layer cannot read, aggregate, build features, write or validate data."""


# ----------------------------------------------------------------------
# Gold aggregator class
# ----------------------------------------------------------------------
class GoldAggregator:
    """Reads the Silver table and builds the daily aggregate table per product and city."""

    def __init__(self, spark, source_table, target_table, logger):
        self.spark = spark
        self.source_table = source_table
        self.target_table = target_table
        self.logger = logger

    def read_silver(self):
        """Read the cleaned Silver table."""
        try:
            self.logger.info("Reading Silver table: %s", self.source_table)
            df = self.spark.table(self.source_table)
            self.logger.info("Silver rows read: %d", df.count())
            return df
        except Exception as e:
            self.logger.error("Failed to read Silver table: %s", e)
            raise GoldError(f"Reading {self.source_table} failed: {e}") from e

    def aggregate(self, df):
        """Group by date, product and city and calculate the four aggregate columns."""
        try:
            agg_df = (
                df.groupBy("transaction_date", "product_name", "destination_city")
                .agg(
                    F.sum("demand_quantity").alias("total_demand"),
                    F.avg("unit_price_usd").alias("avg_unit_price"),
                    F.sum("available_inventory").alias("total_inventory"),
                    F.count("*").alias("transaction_count"),
                )
            )
            self.logger.info("Aggregated rows: %d", agg_df.count())
            return agg_df
        except Exception as e:
            self.logger.error("Failed to aggregate data: %s", e)
            raise GoldError(f"Aggregation failed: {e}") from e

    def write_target(self, df):
        """Write the aggregate DataFrame to the Gold Delta table."""
        try:
            self.logger.info("Writing to Gold aggregate table: %s", self.target_table)
            (
                df.write.format("delta")
                .mode("overwrite")
                .option("overwriteSchema", "true")
                .saveAsTable(self.target_table)
            )
            self.logger.info("Write completed: %s", self.target_table)
        except Exception as e:
            self.logger.error("Failed to write Gold aggregate table: %s", e)
            raise GoldError(f"Writing {self.target_table} failed: {e}") from e

    def validate(self, expected_rows, source_rows):
        """Check the saved aggregate table: row count, unique grain and transaction totals."""
        gold = self.spark.table(self.target_table)
        row_count = gold.count()
        distinct_keys = (
            gold.select("transaction_date", "product_name", "destination_city")
            .distinct()
            .count()
        )
        # Every Silver transaction lands in exactly one group, so this sum must equal the Silver row count.
        # .collect()[0][0] pulls the single result out of the one-row DataFrame as a plain Python number.
        total_transactions = gold.agg(F.sum("transaction_count")).collect()[0][0]
        self.logger.info(
            "Gold check -> rows: %d | distinct keys: %d | total transactions: %d",
            row_count, distinct_keys, total_transactions,
        )
        if row_count != expected_rows:
            raise GoldError(f"Row count mismatch: expected={expected_rows}, table={row_count}")
        if distinct_keys != row_count:
            raise GoldError("Duplicate date/product/city combinations found in Gold table")
        if total_transactions != source_rows:
            raise GoldError(
                f"Transaction total mismatch: silver={source_rows}, gold={total_transactions}"
            )
        return row_count

    def run(self):
        """Run the aggregate pipeline: read -> aggregate -> write -> validate."""
        try:
            self.logger.info("Starting Gold aggregation")
            silver_df = self.read_silver()
            source_rows = silver_df.count()
            agg_df = self.aggregate(silver_df)
            expected_rows = agg_df.count()
            self.write_target(agg_df)
            self.validate(expected_rows, source_rows)
            self.logger.info("Gold aggregation completed successfully")
        except GoldError:
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error in Gold aggregation: {e}")
            raise GoldError(f"Gold aggregation failed: {e}") from e


# ----------------------------------------------------------------------
# Gold feature builder class
# ----------------------------------------------------------------------
class GoldFeatureBuilder:
    """Reads the Gold aggregate table, fills missing days and adds lag, rolling and calendar features."""

    def __init__(self, spark, source_table, target_table, logger):
        self.spark = spark
        self.source_table = source_table
        self.target_table = target_table
        self.logger = logger

    def read_aggregate(self):
        """Read the Gold aggregate table."""
        try:
            self.logger.info("Reading Gold aggregate table: %s", self.source_table)
            df = self.spark.table(self.source_table)
            self.logger.info("Aggregate rows read: %d", df.count())
            return df
        except Exception as e:
            self.logger.error("Failed to read Gold aggregate table: %s", e)
            raise GoldError(f"Reading {self.source_table} failed: {e}") from e

    def build_complete_days(self, df):
        """Add a row for every missing day per product and city, with zero demand."""
        try:
            date_spine = (
                df.groupBy("product_name", "destination_city")
                .agg(
                    F.min("transaction_date").alias("first_date"),
                    F.max("transaction_date").alias("last_date"),
                )
                .withColumn(
                    "transaction_date",
                    F.explode(F.sequence("first_date", "last_date", F.expr("interval 1 day"))),
                )
                .drop("first_date", "last_date")
            )

            complete_df = (
                date_spine.join(
                    df, ["transaction_date", "product_name", "destination_city"], "left"
                )
                .fillna({"total_demand": 0, "transaction_count": 0})
            )
            self.logger.info("Missing days filled with zero demand")
            return complete_df
        except Exception as e:
            self.logger.error("Failed to fill missing days: %s", e)
            raise GoldError(f"Filling missing days failed: {e}") from e

    def add_features(self, df):
        """Add demand_1, demand_7, rolling_7_day_avg, month and day_of_week."""
        try:
            base_window = (
                Window.partitionBy("product_name", "destination_city")
                .orderBy("transaction_date")
            )

            features_df = (
                df.withColumn("demand_1", F.lag("total_demand", 1).over(base_window))
                .withColumn("demand_7", F.lag("total_demand", 7).over(base_window))
                .withColumn(
                    "rolling_7_day_avg",
                    F.avg("total_demand").over(base_window.rowsBetween(-7, -1)),
                )
                .withColumn("month", F.month("transaction_date"))
                .withColumn("day_of_week", F.dayofweek("transaction_date"))
            )
            self.logger.info("Feature columns added: demand_1, demand_7, rolling_7_day_avg, month, day_of_week")
            return features_df
        except Exception as e:
            self.logger.error("Failed to add features: %s", e)
            raise GoldError(f"Feature creation failed: {e}") from e

    def write_target(self, df):
        """Write the feature DataFrame to the Gold feature Delta table."""
        try:
            self.logger.info("Writing to Gold feature table: %s", self.target_table)
            (
                df.write.format("delta")
                .mode("overwrite")
                .option("overwriteSchema", "true")
                .saveAsTable(self.target_table)
            )
            self.logger.info("Write completed: %s", self.target_table)
        except Exception as e:
            self.logger.error("Failed to write Gold feature table: %s", e)
            raise GoldError(f"Writing {self.target_table} failed: {e}") from e

    def validate(self, expected_rows):
        """Check the saved feature table: row count, unique keys, feature columns and nulls."""
        features = self.spark.table(self.target_table)
        row_count = features.count()
        distinct_keys = (
            features.select("transaction_date", "product_name", "destination_city")
            .distinct()
            .count()
        )
        required_columns = ["demand_1", "demand_7", "rolling_7_day_avg", "month", "day_of_week"]
        # Collects any of the five features that didn't get saved. It should be an empty list.
        missing_columns = [c for c in required_columns if c not in features.columns]
        null_keys = features.filter(
            F.col("transaction_date").isNull()
            | F.col("product_name").isNull()
            | F.col("destination_city").isNull()
        ).count()
        null_calendar = features.filter(
            F.col("month").isNull() | F.col("day_of_week").isNull()
        ).count()
        self.logger.info(
            "Feature check -> rows: %d | distinct keys: %d | null keys: %d | null calendar values: %d",
            row_count, distinct_keys, null_keys, null_calendar,
        )
        if row_count != expected_rows:
            raise GoldError(f"Row count mismatch: expected={expected_rows}, table={row_count}")
        if distinct_keys != row_count:
            raise GoldError("Duplicate date/product/city combinations found in feature table")
        if missing_columns:
            raise GoldError(f"Missing feature columns: {missing_columns}")
        if null_keys > 0:
            raise GoldError(f"{null_keys} rows have a null key column")
        if null_calendar > 0:
            raise GoldError(f"{null_calendar} rows have null month or day_of_week")
        return row_count

    def log_null_summary(self):
        """Log how many nulls the three lag/rolling features have (expected: the first rows of each group)."""
        counts = self.spark.table(self.target_table).select(
            F.count(F.when(F.col("demand_1").isNull(), 1)).alias("null_demand_1"),
            F.count(F.when(F.col("demand_7").isNull(), 1)).alias("null_demand_7"),
            F.count(F.when(F.col("rolling_7_day_avg").isNull(), 1)).alias("null_rolling_7_day_avg"),
        ).first()
        self.logger.info(
            "Null counts -> demand_1: %d | demand_7: %d | rolling_7_day_avg: %d",
            counts["null_demand_1"], counts["null_demand_7"], counts["null_rolling_7_day_avg"],
        )

    def run(self):
        """Run the feature pipeline: read -> fill missing days -> add features -> write -> validate."""
        try:
            self.logger.info("Starting Gold feature build")
            agg_df = self.read_aggregate()
            complete_df = self.build_complete_days(agg_df)
            expected_rows = complete_df.count()
            self.logger.info("Rows after filling missing days: %d", expected_rows)
            features_df = self.add_features(complete_df)
            self.write_target(features_df)
            self.validate(expected_rows)
            self.log_null_summary()
            self.logger.info("Gold feature build completed successfully")
        except GoldError:
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error in Gold feature build: {e}")
            raise GoldError(f"Gold feature build failed: {e}") from e


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
def main():
    """Create the Spark session, run both Gold steps in order, and fail loudly on error."""
    spark = SparkSession.builder.getOrCreate()
    try:
        aggregator = GoldAggregator(
            spark=spark,
            source_table=SOURCE_TABLE,
            target_table=AGG_TABLE,
            logger=logger,
        )
        aggregator.run()

        builder = GoldFeatureBuilder(
            spark=spark,
            source_table=AGG_TABLE,
            target_table=FEATURE_TABLE,
            logger=logger,
        )
        builder.run()
    except GoldError as e:
        logger.error(f"Gold pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()