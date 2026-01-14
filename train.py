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

# If you have these modules locally, keep them. 
# If not, comment them out for testing.
from src.preprocessing import load_and_clean_data
from src.config import CATEGORICAL_COLS, NUMERIC_COLS, LABEL_COL

# Experiment parameters
TRAIN_RATIO = 0.8
TEST_RATIO = 0.2
SEED = 42

# Logistic Regression hyperparameters
MAX_ITER = 50
REG_PARAM = 0.01
ELASTIC_NET_PARAM = 0.0

# ===============================
# 2️⃣ START SPARK SESSION
# ===============================
spark = SparkSession.builder \
    .appName("ChurnTraining") \
    .getOrCreate()

# ===============================
# 3️⃣ LOAD AND CLEAN DATA
# ===============================
df = load_and_clean_data(spark)

print("First 5 rows of data:")
df.show(5)

# ===============================
# 4️⃣ SPLIT DATA INTO TRAIN & TEST
# ===============================
train_df, test_df = df.randomSplit([TRAIN_RATIO, TEST_RATIO], seed=SEED)

# ===============================
# 5️⃣ FEATURE ENGINEERING
# ===============================
indexers = [
    StringIndexer(
        inputCol=col,
        outputCol=f"{col}_idx",
        handleInvalid="keep"
    )
    for col in CATEGORICAL_COLS
]

encoders = [
    OneHotEncoder(
        inputCol=f"{col}_idx",
        outputCol=f"{col}_ohe"
    )
    for col in CATEGORICAL_COLS
]

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
lr = LogisticRegression(
    labelCol=LABEL_COL,
    featuresCol="features",
    maxIter=MAX_ITER,
    regParam=REG_PARAM,
    elasticNetParam=ELASTIC_NET_PARAM
)

# ===============================
# 7️⃣ BUILD PIPELINE
# ===============================
pipeline = Pipeline(
    stages=indexers + encoders + [assembler, lr]
)

# ===============================
# 8️⃣ EVALUATOR
# ===============================
evaluator = BinaryClassificationEvaluator(
    labelCol=LABEL_COL,
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)

# ===============================
# 9️⃣ HYPERPARAMETER GRID
# ===============================
paramGrid = (
    ParamGridBuilder()
    .addGrid(lr.regParam, [0.01, 0.1])
    .addGrid(lr.elasticNetParam, [0.0, 1.0])
    .build()
)

# ===============================
# 🔟 CROSS VALIDATION
# ===============================
crossval = CrossValidator(
    estimator=pipeline,
    estimatorParamMaps=paramGrid,
    evaluator=evaluator,
    numFolds=3
)

# ===============================
# 1️⃣1️⃣ TRAIN + TRACK WITH MLFLOW
# ===============================

# CHANGE 1: Capture the run object using 'as run'
with mlflow.start_run() as run:

    # Log experiment parameters
    mlflow.log_param("model_type", "logistic_regression")
    mlflow.log_param("train_ratio", TRAIN_RATIO)
    mlflow.log_param("test_ratio", TEST_RATIO)
    mlflow.log_param("seed", SEED)

    # Log model hyperparameters
    mlflow.log_param("maxIter", MAX_ITER)
    mlflow.log_param("regParam", REG_PARAM)
    mlflow.log_param("elasticNetParam", ELASTIC_NET_PARAM)

    # ---- Training ----
    print("Training model...")
    cv_model = crossval.fit(train_df)

    # Make predictions
    predictions = cv_model.bestModel.transform(test_df)
    
    # Evaluate
    auc = evaluator.evaluate(predictions)
    print(f"Test AUC: {auc:.4f}")
    mlflow.log_metric("test_auc", auc)

    # Log Model with Signature
   # ---------------------------------------------------------
    # ---------------------------------------------------------
    
    # 1. Define exactly what columns the model needs (From your config)
    required_features = CATEGORICAL_COLS + NUMERIC_COLS
    
    # 2. Select ONLY those columns for the example
    # We deliberately EXCLUDE 'label', 'Churn', and 'customerID'
    input_example_pd = test_df.select(required_features).limit(5).toPandas()

    # 3. Log the model with the CLEAN example
    mlflow.spark.log_model(
        spark_model=cv_model.bestModel,
        artifact_path="logistic_regression_pipeline",
        input_example=input_example_pd 
    )
    
    print("Model logged successfully with CLEAN signature (Features Only)!")
    
    print("Model logged successfully with signature!")

# ===============================
# 1️⃣2️⃣ REGISTER MODEL (OUTSIDE THE RUN)
# ===============================

# CHANGE 2: Use the 'run' variable we captured earlier
# We don't need 'active_run()' because we have 'run.info.run_id' saved.

print("Registering the model...")

model_uri = f"runs:/{run.info.run_id}/logistic_regression_pipeline"
model_name = "Churn_Prediction_Model_Brian"

model_details = mlflow.register_model(model_uri, model_name)

print(f"Model registered! Version: {model_details.version}")

# Optional: Transition to Staging
from mlflow.tracking import MlflowClient
client = MlflowClient()

client.transition_model_version_stage(
    name=model_name,
    version=model_details.version,
    stage="Staging"
)

print(f"Model Version {model_details.version} moved to Staging.")

# ===============================
# 🛑 STOP SPARK
# ===============================
spark.stop()