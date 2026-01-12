from pyspark.sql import SparkSession
from src.preprocessing import load_and_clean_data

spark = SparkSession.builder \
    .appName("TestPreprocessing") \
    .getOrCreate()

df = load_and_clean_data(spark)

df.show(5)
print(f"Rows: {df.count()}")

spark.stop()
