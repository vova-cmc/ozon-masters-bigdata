import os, sys

SPARK_HOME = "/usr/hdp/current/spark2-client"
PYSPARK_PYTHON = "/opt/conda/envs/dsenv/bin/python"
os.environ["PYSPARK_PYTHON"]= PYSPARK_PYTHON
os.environ["SPARK_HOME"] = SPARK_HOME

PYSPARK_HOME = os.path.join(SPARK_HOME, "python/lib")
sys.path.insert(0, os.path.join(PYSPARK_HOME, "py4j-0.10.7-src.zip"))
sys.path.insert(0, os.path.join(PYSPARK_HOME, "pyspark.zip"))

from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer, CountVectorizer
from pyspark.ml.regression import LinearRegression

tokenizer = RegexTokenizer(minTokenLength=2, pattern='\\W', inputCol="reviewText", outputCol="words")
cv = CountVectorizer(vocabSize=5 * 10e3, inputCol=tokenizer.getOutputCol(), outputCol="cv")
lr = LinearRegression(featuresCol=cv.getOutputCol(), labelCol="overall", maxIter=25)

pipeline = Pipeline(stages=[
    tokenizer,
    cv,
    lr
])