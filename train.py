# ===============================
# 1️⃣ IMPORT LIBRARIES
# ===============================

from pyspark.sql import SparkSession

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

import mlflow
import mlflow.spark

from src.preprocessing import load_and_clean_data
from src.config import CATEGORICAL_COLS, NUMERIC_COLS, LABEL_COL


# ===============================
# 2️⃣ START SPARK SESSION
# ===============================
# SparkSession is the entry point to Spark

spark = SparkSession.builder \
    .appName("ChurnTraining") \
    .getOrCreate()


# ===============================
# 3️⃣ LOAD AND CLEAN DATA
# ===============================
# This function returns a Spark DataFrame

df = load_and_clean_data(spark)

print("First 5 rows of data:")
df.show(5)


# ===============================
# 4️⃣ SPLIT DATA INTO TRAIN & TEST
# ===============================
# 80% for training, 20% for testing

train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)


# ===============================
# 5️⃣ FEATURE ENGINEERING
# ===============================

# Convert categorical columns (strings) → numbers
indexers = [
    StringIndexer(
        inputCol=col,
        outputCol=f"{col}_idx",
        handleInvalid="keep"   # unseen values go to "unknown"
    )
    for col in CATEGORICAL_COLS
]

# One-hot encode indexed categories
encoders = [
    OneHotEncoder(
        inputCol=f"{col}_idx",
        outputCol=f"{col}_ohe"
    )
    for col in CATEGORICAL_COLS
]

# Combine categorical + numeric columns
feature_cols = (
    [f"{col}_ohe" for col in CATEGORICAL_COLS] +
    NUMERIC_COLS
)

assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features"
)


# ===============================
# 6️⃣ DEFINE MODEL
# ===============================
# Logistic Regression for binary classification

lr = LogisticRegression(
    featuresCol="features",
    labelCol=LABEL_COL,
    maxIter=20
)

# ===============================
# 7️⃣ BUILD PIPELINE
# ===============================
# Pipeline ensures all steps run in the correct order

pipeline = Pipeline(
    stages=indexers + encoders + [assembler, lr]
)

# ===============================
# 8️⃣ EVALUATOR
# ===============================
# AUC measures how well the model separates churn vs no-churn

evaluator = BinaryClassificationEvaluator(
    labelCol=LABEL_COL,
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)

# ===============================
# 9️⃣ HYPERPARAMETER GRID
# ===============================
# We try different values to find the best model

paramGrid = (
    ParamGridBuilder()
    .addGrid(lr.regParam, [0.01, 0.1])        # regularization strength
    .addGrid(lr.elasticNetParam, [0.0, 1.0]) # L2 vs L1
    .build()
)

# ===============================
# 🔟 CROSS VALIDATION
# ===============================
# Spark will train multiple models and pick the best one

crossval = CrossValidator(
    estimator=pipeline,
    estimatorParamMaps=paramGrid,
    evaluator=evaluator,
    numFolds=3
)


# ===============================
# 1️⃣1️⃣ TRAIN + TRACK WITH MLFLOW
# ===============================

with mlflow.start_run():

    # Train model using cross-validation
    cv_model = crossval.fit(train_df)

    # Make predictions on test data
    predictions = cv_model.bestModel.transform(test_df)

    # Evaluate model
    auc = evaluator.evaluate(predictions)
    print(f"Test AUC: {auc:.4f}")

    # Log metrics to MLflow
    mlflow.log_metric("test_auc", auc)

    # Save model to MLflow
    mlflow.spark.log_model(
        cv_model.bestModel,
        "logistic_regression_pipeline"
    )
    spark.stop()
# ===============================
