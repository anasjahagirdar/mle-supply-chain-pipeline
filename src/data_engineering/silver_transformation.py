"""Silver transformation: remove duplicates and fix the date type, then store the clean Silver table."""

import sys
import logging

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
SOURCE_TABLE = "mle_project.bronze.supply_chain_raw"
TARGET_TABLE = "mle_project.silver.supply_chain_clean"


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


logger = get_logger("silver_transformation")


class TransformationError(Exception):
    """Raised when the Silver layer cannot read, transform, write or validate data."""


# ----------------------------------------------------------------------
# Silver transformer class
# ----------------------------------------------------------------------
class SilverTransformer:
    """Reads the Bronze table, removes duplicates, fixes the date type and writes the Silver table."""

    def __init__(self, spark, source_table, target_table, logger):
        self.spark = spark
        self.source_table = source_table
        self.target_table = target_table
        self.logger = logger

    def read_bronze(self):
        """Read the raw Bronze table."""
        try:
            self.logger.info("Reading Bronze table: %s", self.source_table)
            df = self.spark.table(self.source_table)
            self.logger.info("Bronze rows read: %d", df.count())
            return df
        except Exception as e:
            self.logger.error("Failed to read Bronze table: %s", e)
            raise TransformationError(f"Reading {self.source_table} failed: {e}") from e

    def remove_duplicates(self, df):
        """Keep exactly one row per transaction_id."""
        try:
            rows_before = df.count()
            deduped = df.dropDuplicates(["transaction_id"])
            rows_after = deduped.count()
            self.logger.info(
                "Duplicates removed: %d (rows before: %d, rows after: %d)",
                rows_before - rows_after, rows_before, rows_after,
            )
            return deduped
        except Exception as e:
            self.logger.error("Failed to remove duplicates: %s", e)
            raise TransformationError(f"Duplicate removal failed: {e}") from e

    def convert_date(self, df):
        """Convert transaction_date from string to a real date type."""
        try:
            converted = df.withColumn(
                "transaction_date",
                F.to_date(F.col("transaction_date"), "yyyy-MM-dd"),
            )
            null_dates = converted.filter(F.col("transaction_date").isNull()).count()
            if null_dates > 0:
                raise TransformationError(f"{null_dates} rows have an invalid transaction_date")
            self.logger.info("transaction_date converted to date type")
            return converted
        except TransformationError:
            raise
        except Exception as e:
            self.logger.error("Failed to convert transaction_date: %s", e)
            raise TransformationError(f"Date conversion failed: {e}") from e

    def write_target(self, df):
        """Write the cleaned DataFrame to the Silver Delta table."""
        try:
            self.logger.info("Writing to Silver table: %s", self.target_table)
            (
                df.write.format("delta")
                .mode("overwrite")
                .option("overwriteSchema", "true")
                .saveAsTable(self.target_table)
            )
            self.logger.info("Write completed: %s", self.target_table)
        except Exception as e:
            self.logger.error("Failed to write Silver table: %s", e)
            raise TransformationError(f"Writing {self.target_table} failed: {e}") from e

    def validate(self, expected_count):
        """Check the saved Silver table: row count, unique IDs and date type."""
        silver = self.spark.table(self.target_table)
        row_count = silver.count()
        distinct_ids = silver.select("transaction_id").distinct().count()
        date_type = dict(silver.dtypes)["transaction_date"]
        self.logger.info(
            "Silver check -> rows: %d | distinct ids: %d | transaction_date type: %s",
            row_count, distinct_ids, date_type,
        )
        if row_count != expected_count:
            raise TransformationError(
                f"Row count mismatch: expected={expected_count}, table={row_count}"
            )
        if distinct_ids != row_count:
            raise TransformationError("Duplicate transaction_id values found in Silver table")
        if date_type != "date":
            raise TransformationError(f"transaction_date has type {date_type}, expected date")
        return row_count

    def log_date_range(self):
        """Log the first and last transaction_date in the Silver table."""
        dates = self.spark.table(self.target_table).select(
            F.min("transaction_date").alias("min_date"),
            F.max("transaction_date").alias("max_date"),
        ).first()
        self.logger.info("Date range -> min: %s | max: %s", dates["min_date"], dates["max_date"])

    def run(self):
        """Run the full Silver pipeline: read -> dedupe -> convert date -> write -> validate."""
        try:
            self.logger.info("Starting Silver transformation")
            bronze_df = self.read_bronze()
            deduped_df = self.remove_duplicates(bronze_df)
            converted_df = self.convert_date(deduped_df)
            expected_count = converted_df.count()
            self.write_target(converted_df)
            self.validate(expected_count)
            self.log_date_range()
            self.logger.info("Silver transformation completed successfully")
        except TransformationError:
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error in Silver pipeline: {e}")
            raise TransformationError(f"Silver pipeline failed: {e}") from e


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
def main():
    """Create the Spark session, run the Silver transformation, and fail loudly on error."""
    spark = SparkSession.builder.getOrCreate()
    try:
        transformer = SilverTransformer(
            spark=spark,
            source_table=SOURCE_TABLE,
            target_table=TARGET_TABLE,
            logger=logger,
        )
        transformer.run()
    except TransformationError as e:
        logger.error(f"Silver transformation failed: {e}")
        raise


if __name__ == "__main__":
    main()