"""Bronze ingestion: read the raw supply chain CSV and store it, unchanged, as a Delta table."""

import sys
import logging

from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField,
    StringType, IntegerType, DoubleType,
)

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
SOURCE_PATH  = "/Volumes/mle_project/bronze/raw_files/supply_chain_dataset_wrong.csv"
TARGET_TABLE = "mle_project.bronze.supply_chain_raw"


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


logger = get_logger("bronze_ingestion")


class IngestionError(Exception):
    """Raised when raw data cannot be read, written or validated in the Bronze layer."""


# ----------------------------------------------------------------------
# Schema (explicit, so data types are predictable)
# ----------------------------------------------------------------------
SUPPLY_CHAIN_SCHEMA = StructType([
    StructField("transaction_id",              StringType(),  True),
    StructField("transaction_date",            StringType(),  True),
    StructField("transaction_year",            IntegerType(), True),
    StructField("transaction_quarter",         IntegerType(), True),
    StructField("transaction_month",           IntegerType(), True),
    StructField("product_name",                StringType(),  True),
    StructField("product_category",            StringType(),  True),
    StructField("quantity_unit",               StringType(),  True),
    StructField("supplier_name",               StringType(),  True),
    StructField("supplier_country",            StringType(),  True),
    StructField("supplier_reliability_score",  DoubleType(),  True),
    StructField("refinery_name",               StringType(),  True),
    StructField("destination_city",            StringType(),  True),
    StructField("transportation_mode",         StringType(),  True),
    StructField("ordered_quantity",            DoubleType(),  True),
    StructField("demand_quantity",             DoubleType(),  True),
    StructField("available_inventory",         DoubleType(),  True),
    StructField("unit_price_usd",              DoubleType(),  True),
    StructField("product_cost_usd",            DoubleType(),  True),
    StructField("transportation_cost_usd",     DoubleType(),  True),
    StructField("total_cost_usd",              DoubleType(),  True),
    StructField("expected_lead_time_days",     IntegerType(), True),
    StructField("actual_lead_time_days",       IntegerType(), True),
    StructField("delay_days",                  IntegerType(), True),
    StructField("is_delayed",                  IntegerType(), True),
    StructField("is_stockout",                 IntegerType(), True),
    StructField("quality_status",              StringType(),  True),
    StructField("quality_score",               DoubleType(),  True),
    StructField("disruption_type",             StringType(),  True),
    StructField("delivery_status",             StringType(),  True),
    StructField("ingestion_timestamp",         StringType(),  True),
    StructField("source_system",               StringType(),  True),
    StructField("operation_type",              StringType(),  True),
])


# ----------------------------------------------------------------------
# Ingestion class
# ----------------------------------------------------------------------
class BronzeIngestor:
    """Reads a raw CSV and stores it, unchanged, as a Delta table."""

    def __init__(self, spark, source_path, target_table, schema, logger):
        self.spark = spark
        self.source_path = source_path
        self.target_table = target_table
        self.schema = schema
        self.logger = logger

    def read_source(self):
        """Read the CSV using the explicit schema."""
        try:
            self.logger.info("Reading source file: %s", self.source_path)
            df = (
                self.spark.read.format("csv")
                .option("header", "true")
                .option("mode", "FAILFAST")
                .schema(self.schema)
                .load(self.source_path)
            )
            self.logger.info("Source has %d columns", len(df.columns))
            return df
        except Exception as e:
            self.logger.error("Failed to read source: %s", e)
            raise IngestionError(f"Reading {self.source_path} failed: {e}") from e

    def write_target(self, df):
        """Write the DataFrame to a Delta table (overwrite, so re-runs are safe)."""
        try:
            self.logger.info("Writing to Delta table: %s", self.target_table)
            (
                df.write.format("delta")
                .mode("overwrite")
                .option("overwriteSchema", "true")
                .saveAsTable(self.target_table)
            )
            self.logger.info("Write completed: %s", self.target_table)
        except Exception as e:
            self.logger.error("Failed to write table: %s", e)
            raise IngestionError(f"Writing {self.target_table} failed: {e}") from e

    def validate(self, source_count):
        """Check that the table holds exactly as many rows as the source file."""
        target_count = self.spark.table(self.target_table).count()
        self.logger.info("Row count -> source: %d | bronze table: %d", source_count, target_count)
        if source_count != target_count:
            raise IngestionError(
                f"Row count mismatch: source={source_count}, table={target_count}"
            )
        return target_count

    def log_duplicate_summary(self):
        """Log how many duplicate rows Bronze holds (Bronze is raw, so duplicates are expected)."""
        bronze_df = self.spark.table(self.target_table)
        total_rows = bronze_df.count()
        distinct_rows = bronze_df.dropDuplicates().count()
        self.logger.info(
            "Total rows: %d | Distinct rows: %d | Duplicates: %d",
            total_rows, distinct_rows, total_rows - distinct_rows,
        )

    def run(self):
        """Run the full ingestion: read -> count -> write -> validate -> duplicate summary."""
        self.logger.info("=== Bronze ingestion started ===")
        df = self.read_source()
        source_count = df.count()
        self.write_target(df)
        self.validate(source_count)
        self.log_duplicate_summary()
        self.logger.info("=== Bronze ingestion finished successfully ===")


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
def main():
    """Create the Spark session, run the ingestion, and fail loudly on error."""
    spark = SparkSession.builder.getOrCreate()
    try:
        ingestor = BronzeIngestor(
            spark=spark,
            source_path=SOURCE_PATH,
            target_table=TARGET_TABLE,
            schema=SUPPLY_CHAIN_SCHEMA,
            logger=logger,
        )
        ingestor.run()
    except IngestionError:
        logger.error("Bronze ingestion failed. See the messages above.")
        raise


if __name__ == "__main__":
    main()
