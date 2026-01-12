from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark import SparkFiles

from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression


# 1. Create SparkSession

spark = SparkSession.builder \
    .appName("ChurnPreprocessing") \
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
    inferSchema=True)

# ... (your previous code)

# ADD THESE LINES TO SEE THE DATA:
print("Successfully processed the data. Here are the first 5 rows:")
df.show(5)

print("Here is the confirmed Schema:")
#df.printSchema()

print(f"Total row count: {df.count()}")

null_counts = df.select([
    col(c).isNull().cast("int").alias(c)
    for c in df.columns
])

df.select("TotalCharges").show(20, truncate=False)

null_counts.show()

#df = df.withColumn(
 #   "TotalCharges",
  #  col("TotalCharges").cast("double")
#)

# Replace your current Line 44 block with this:
df = df.withColumn(
    "TotalCharges",
    when(col("TotalCharges") == " ", None) # Safety check: Space becomes Null
    .otherwise(col("TotalCharges"))       # Otherwise keep it
    .cast("double")                        # Now it's safe to turn into a number
)

# This count will finally work and show 7032!
print(f"Final row count: {df.count()}")



df.select("TotalCharges").printSchema()

df = df.dropna(subset=["TotalCharges"])

print("\n--- AFTER CLEANING ---")
print(f"Final row count: {df.count()}") # This will now show 7032


df = df.withColumn(
    "label",
    when(col("Churn") == "Yes", 1).otherwise(0)
)

df.select("Churn", "label").show(5)
df.printSchema()
print(f"Rows after cleaning: {df.count()}")

# ================================
# STEP 4: FEATURE ENGINEERING
# ================================

# Target column
label_col = "label"

# Categorical columns
categorical_cols = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod"
]

# Numerical columns
numeric_cols = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

indexers = [
    StringIndexer(
        inputCol=col,
        outputCol=f"{col}_indexed",
        handleInvalid="keep"
    )
    for col in categorical_cols
]


encoder = OneHotEncoder(
    inputCols=[f"{col}_indexed" for col in categorical_cols],
    outputCols=[f"{col}_encoded" for col in categorical_cols]
)


assembler = VectorAssembler(
    inputCols=numeric_cols + [f"{col}_encoded" for col in categorical_cols],
    outputCol="features"
)

pipeline = Pipeline(
    stages=indexers + [encoder, assembler]
)

feature_df = pipeline.fit(df).transform(df) 

feature_df.select("features", "label").show(5, truncate=False)

first_row = feature_df.select("features").first()
print(f"Feature vector size: {len(first_row['features'])}")
feature_df.printSchema()

# ================================
# STEP 5: TRAIN / TEST SPLIT
# ================================

train_df, test_df = feature_df.randomSplit(
    [0.8, 0.2],
    seed=42
)

print(f"Training rows: {train_df.count()}")
print(f"Test rows: {test_df.count()}")

#create Logistic Regression model
lr = LogisticRegression(
    featuresCol="features",
    labelCol="label",
    maxIter=50,
    regParam=0.01
)

lr_model = lr.fit(train_df)

predictions = lr_model.transform(test_df)

predictions.select(
    "label",
    "prediction",
    "probability"
).show(10, truncate=False)
#breakpoint()






spark.stop()