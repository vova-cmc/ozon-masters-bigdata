#!/bin/bash
./train.py /datasets/amazon/all_reviews_5_core_train.json pipeline_model_task_4.mdl
./predict.py pipeline_model_task_4.mdl /datasets/amazon/all_reviews_5_core_test_features.json predictions.json