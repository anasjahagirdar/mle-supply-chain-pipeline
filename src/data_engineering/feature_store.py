"""Feature store: turn the Gold feature table into a Unity Catalog feature table by adding a primary key."""

import sys
import logging

from pyspark.sql import SparkSession

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
FEATURE_TABLE       = "mle_project.gold.supply_chain_features"
PRIMARY_KEY_COLUMNS = ["transaction_date", "product_name", "destination_city"]
PRIMARY_KEY_NAME    = "supply_chain_features_pk"


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


logger = get_logger("feature_store")


class FeatureStoreError(Exception):
    """Raised when the feature table cannot be prepared, registered or validated."""


# ----------------------------------------------------------------------
# Feature table preparer class
# ----------------------------------------------------------------------
class FeatureTablePreparer:
    """Turns the Gold feature table into a Unity Catalog feature table by adding a primary key."""

    def __init__(self, spark, table_name, primary_key_columns, primary_key_name, logger):
        self.spark = spark
        self.table_name = table_name
        self.primary_key_columns = primary_key_columns
        self.primary_key_name = primary_key_name
        self.logger = logger
        self.catalog_name, self.schema_name, self.short_table_name = table_name.split(".")

    def make_key_columns_not_null(self):
        """Primary key columns must be NOT NULL, so set that on each key column."""
        try:
            for column in self.primary_key_columns:
                self.spark.sql(f"ALTER TABLE {self.table_name} ALTER COLUMN {column} SET NOT NULL")
                self.logger.info("Column set to NOT NULL: %s", column)
        except Exception as e:
            self.logger.error("Failed to set key columns to NOT NULL: %s", e)
            raise FeatureStoreError(f"Setting NOT NULL on key columns failed: {e}") from e

    def add_primary_key(self):
        """Add the composite primary key. Any old primary key is dropped first so the step can be re-run."""
        try:
            key_columns = ", ".join(self.primary_key_columns)
            self.spark.sql(f"ALTER TABLE {self.table_name} DROP PRIMARY KEY IF EXISTS")
            self.spark.sql(
                f"ALTER TABLE {self.table_name} "
                f"ADD CONSTRAINT {self.primary_key_name} PRIMARY KEY ({key_columns})"
            )
            self.logger.info("Primary key %s added on: %s", self.primary_key_name, key_columns)
        except Exception as e:
            self.logger.error("Failed to add primary key: %s", e)
            raise FeatureStoreError(f"Adding primary key to {self.table_name} failed: {e}") from e

    def validate(self):
        """Check the primary key exists, the key columns are NOT NULL and the keys are unique."""
        key_rows = self.spark.sql(f"""
            SELECT column_name
            FROM {self.catalog_name}.information_schema.key_column_usage
            WHERE table_schema = '{self.schema_name}'
              AND table_name = '{self.short_table_name}'
              AND constraint_name = '{self.primary_key_name}'
            ORDER BY ordinal_position
        """).collect()
        saved_key_columns = [row["column_name"] for row in key_rows]

        table_df = self.spark.table(self.table_name)
        nullable_key_columns = [
            field.name
            for field in table_df.schema.fields
            if field.name in self.primary_key_columns and field.nullable
        ]

        row_count = table_df.count()
        distinct_keys = table_df.select(*self.primary_key_columns).distinct().count()

        self.logger.info(
            "Feature table check -> primary key columns: %s | rows: %d | distinct keys: %d",
            saved_key_columns, row_count, distinct_keys,
        )
        if saved_key_columns != self.primary_key_columns:
            raise FeatureStoreError(
                f"Primary key mismatch: expected={self.primary_key_columns}, found={saved_key_columns}"
            )
        if nullable_key_columns:
            raise FeatureStoreError(f"Key columns still allow nulls: {nullable_key_columns}")
        if distinct_keys != row_count:
            raise FeatureStoreError("Duplicate keys found in the feature table")
        return row_count

    def run(self):
        """Run the steps: NOT NULL on key columns -> add primary key -> validate."""
        try:
            self.logger.info("Starting feature table preparation: %s", self.table_name)
            self.make_key_columns_not_null()
            self.add_primary_key()
            self.validate()
            self.logger.info("Feature table is ready: %s", self.table_name)
        except FeatureStoreError:
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error in feature table preparation: {e}")
            raise FeatureStoreError(f"Feature table preparation failed: {e}") from e


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
def main():
    """Create the Spark session, prepare the feature table, and fail loudly on error."""
    spark = SparkSession.builder.getOrCreate()
    try:
        preparer = FeatureTablePreparer(
            spark=spark,
            table_name=FEATURE_TABLE,
            primary_key_columns=PRIMARY_KEY_COLUMNS,
            primary_key_name=PRIMARY_KEY_NAME,
            logger=logger,
        )
        preparer.run()
    except FeatureStoreError as e:
        logger.error(f"Feature store step failed: {e}")
        raise


if __name__ == "__main__":
    main()