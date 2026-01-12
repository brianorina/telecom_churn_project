from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("SparkTest") \
    .getOrCreate()

data = [("Alice", 30), ("Bob", 25)]
df = spark.createDataFrame(data, ["name", "age"])

df.show()

spark.stop()
