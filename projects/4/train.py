#!/opt/conda/envs/dsenv/bin/python

import os, sys, datetime
import logging

#
# Logging initialization
#
logging.basicConfig(level=logging.INFO)
logging.info("CURRENT_DIR {}".format(os.getcwd()))
logging.info("SCRIPT CALLED AS {}".format(sys.argv[0]))
logging.info("ARGS {}".format(sys.argv[1:]))

#
# Read script arguments
#
try:
    DATA_PATH = sys.argv[1] 
    OUTPUT_MODEL_PATH = sys.argv[2]
except:
    logging.critical("Not enough params")
    sys.exit(1)
#
# Spark init
#

SPARK_HOME = "/usr/hdp/current/spark2-client"
PYSPARK_PYTHON = "/opt/conda/envs/dsenv/bin/python"
os.environ["PYSPARK_PYTHON"]= PYSPARK_PYTHON
os.environ["SPARK_HOME"] = SPARK_HOME

PYSPARK_HOME = os.path.join(SPARK_HOME, "python/lib")
sys.path.insert(0, os.path.join(PYSPARK_HOME, "py4j-0.10.7-src.zip"))
sys.path.insert(0, os.path.join(PYSPARK_HOME, "pyspark.zip"))

from pyspark import SparkContext, SparkConf
from pyspark.sql import functions as f
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType

spark = SparkSession.builder.getOrCreate()
spark.sparkContext.setLogLevel('WARN')

from model import pipeline

#
# Now do something useful
#

start = datetime.datetime.now()

dataset = spark.read.json(DATA_PATH)
dataset = dataset.drop("image", "reviewerName", "unixReviewTime").cache()

verified = f.when(dataset.verified, 1).otherwise(0)
vote = f.when(dataset.vote.isNull(), 0).otherwise(dataset.vote.astype(IntegerType()))

dataset = dataset.withColumn("verified", verified)
dataset = dataset.withColumn("vote", vote)
dataset = dataset.drop("id", "asin", "reviewTime", "reviewerID", "summary").cache()
logging.info("Finished preparing data")

pipeline_model = pipeline.fit(dataset)
logging.info("Finished fitting the model")
pipeline_model.write().overwrite().save(OUTPUT_MODEL_PATH)
logging.info(f"Model saved to {OUTPUT_MODEL_PATH}")
logging.info(f"Time elapsed: {datetime.datetime.now() - start}")

spark.stop()