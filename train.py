from pyspark.sql import SparkSession

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

from src.preprocessing import load_and_clean_data
from src.config import CATEGORICAL_COLS, NUMERIC_COLS, LABEL_COL

from pyspark.ml.feature import (
    StringIndexer,
    OneHotEncoder,
    VectorAssembler
)


spark = SparkSession.builder \
    .appName("ChurnTraining") \
    .getOrCreate()

df = load_and_clean_data(spark)

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

lr = LogisticRegression(
    featuresCol="features",
    labelCol=LABEL_COL,
    maxIter=20
)

pipeline = Pipeline(
    stages=indexers + encoders + [assembler, lr]
)

train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

model = pipeline.fit(train_df)


predictions = model.transform(test_df)

predictions.select(
    "label",
    "probability",
    "prediction"
).show(10, truncate=False)


evaluator = BinaryClassificationEvaluator(
    labelCol=LABEL_COL,
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)

auc = evaluator.evaluate(predictions)
print(f"AUC: {auc:.4f}")





print("First 5 rows of data:")
df.show(5)

# pipeline code continues...



