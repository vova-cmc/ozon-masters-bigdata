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
    start_point = sys.argv[1] 
    end_point = sys.argv[2]
    data_path = sys.argv[3]
    output_dir = sys.argv[4]
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
 
conf = SparkConf()
sc = SparkContext(appName="BFS", conf=conf)

#
# Now do something useful
#

start = datetime.datetime.now()
raw_graph = sc.textFile(data_path)
# raw_graph.collect()
raw_graph = raw_graph.map(lambda x: x.split('\t'))
raw_graph = raw_graph.map(lambda x: [x[1],x[0]])

points_to_analyse = [start_point]

paths = raw_graph.filter(lambda x: x[0] in points_to_analyse).map(lambda x: [x[1],[x[0]] + [x[1]]])
i = 0
while i < 10:
    logging.info(i)
    points_to_analyse_new = paths.map(lambda x: x[0]).collect()
    logging.debug(points_to_analyse_new)

    if end_point in points_to_analyse_new:
        print('Eurica!')
        break

    new_vertices = raw_graph.filter(lambda x: x[0] in points_to_analyse_new)
    paths = paths.join(new_vertices).map(lambda x: [x[1][1],x[1][0] + [x[1][1]]])
    logging.debug(new_vertices.collect())
    logging.debug(paths.collect())

    i+=1
res = paths.filter(lambda x: x[0] == end_point).collect()
logging.info(f"Time elapsed: {datetime.datetime.now() - start}")

with open('result.csv','w') as f:
    for el in res:
        f.write(','.join(el[1]) + '\n')
os.system('hdfs dfs -mkdir /user/vova-cmc/'+output_dir)
os.system('hdfs dfs -copyFromLocal result.csv /user/vova-cmc/'+output_dir+'/result.csv')
sc.stop()