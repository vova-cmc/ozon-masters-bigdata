from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer, CountVectorizer
from pyspark.ml.regression import LinearRegression

tokenizer = RegexTokenizer(minTokenLength=2, pattern='\\W', inputCol="reviewText", outputCol="words")
cv = CountVectorizer(vocabSize=5 * 10e3, inputCol=tokenizer.getOutputCol(), outputCol="cv")
lr = LinearRegression(featuresCol=cv.getOutputCol(), labelCol="overall", maxIter=25)

pipeline = Pipeline(stages=[
    tokenizer,
    cv_model,
    lr_model
])