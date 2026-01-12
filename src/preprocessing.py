from pyspark.sql.functions import col, when
from pyspark import SparkFiles

from src.config import LABEL_COL


def load_and_clean_data(spark):
    url = "https://raw.githubusercontent.com/treselle-systems/customer_churn_analysis/master/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    spark.sparkContext.addFile(url)

    df = spark.read.csv(
        "file://" + SparkFiles.get("WA_Fn-UseC_-Telco-Customer-Churn.csv"),
        header=True,
        inferSchema=True
    )

    df = df.withColumn(
        "TotalCharges",
        when(col("TotalCharges") == " ", None)
        .otherwise(col("TotalCharges"))
        .cast("double")
    )

    df = df.dropna(subset=["TotalCharges"])

    df = df.withColumn(
        LABEL_COL,
        when(col("Churn") == "Yes", 1).otherwise(0)
    )

    return df
