from pyspark.sql import SparkSession
from pyspark import SparkFiles

# 1. Create SparkSession
spark = SparkSession.builder \
    .appName("ChurnIngestion") \
    .getOrCreate()

# 2. Define the URL
url = "https://raw.githubusercontent.com/treselle-systems/customer_churn_analysis/master/WA_Fn-UseC_-Telco-Customer-Churn.csv"

# 3. Add the file to Spark (this downloads it to a temporary location)
spark.sparkContext.addFile(url)

# 4. Load CSV using SparkFiles.get() to find the temporary path
# The filename in SparkFiles.get() must match the end of the URL
df = spark.read.csv(
    "file://" + SparkFiles.get("WA_Fn-UseC_-Telco-Customer-Churn.csv"), 
    header=True, 
    inferSchema=True
)
print(f"Number of partitions: {df.rdd.getNumPartitions()}")
