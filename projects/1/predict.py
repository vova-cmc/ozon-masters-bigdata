#!/opt/conda/envs/dsenv/bin/python

import os, sys
import logging

import pandas as pd
from sklearn.model_selection import train_test_split
from joblib import dump
from sklearn.metrics import log_loss

#
# Logging initialization
#
logging.basicConfig(level=logging.DEBUG)
logging.info("CURRENT_DIR {}".format(os.getcwd()))
logging.info("SCRIPT CALLED AS {}".format(sys.argv[0]))
logging.info("ARGS {}".format(sys.argv[1:]))

#
# Read script arguments
#
try:
    proj_id = sys.argv[1] 
    train_path = sys.argv[2]
except:
    logging.critical("Need to pass both project_id and train dataset path")
    sys.exit(1)


logging.info(f"TRAIN_ID {proj_id}")
logging.info(f"TRAIN_PATH {train_path}")
# from sklearn import __version__
# logging.info(f"CAT_FEATURES_USED {cat_features_to_use}")

from joblib import load
model = load("1.joblib")

#
# Read dataset
#
read_opts=dict(
        sep=',', names=fields, index_col=False, header=None,
        iterator=True, chunksize=100
)


for df in pd.read_csv(sys.stdin, **read_opts):
    pred = model.predict(df)
    out = zip(df.doc_id, pred)
    print("\n".join(["{0},{1}".format(*i) for i in out]))
