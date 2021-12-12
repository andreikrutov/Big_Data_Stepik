#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
from pyspark.sql import SparkSession

from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.pipeline import PipelineModel
from pyspark.ml.tuning import TrainValidationSplitModel

from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit

# Используйте как путь откуда загрузить модель
MODEL_PATH = 'spark_ml_model'


def process(spark, input_file, output_file):
    
    '''
    Функция принимает на вход путь до test файла и путь к файлу, в коорый записывает результат.
    Она загружает модель из 'spark_ml_model' и при попмощи нее делает предсказания по данным test.
    '''
    df = spark.read.parquet(input_file)
    feature = VectorAssembler(inputCols=df.columns[:-1], outputCol='features')
    feature_vector = feature.transform(df)
    
    model = TrainValidationSplitModel.load('spark_ml_model')
    
    predictions = model.transform(feature_vector)
    predictions = predictions.select('ad_id','prediction')
    predictions.coalesce(1).write.csv(output_file)


def main(argv):
    input_path = argv[0]
    print("Input path to file: " + input_path)
    output_file = argv[1]
    print("Output path to file: " + output_file)
    spark = _spark_session()
    process(spark, input_path, output_file)


def _spark_session():
    return SparkSession.builder.appName('PySparkMLPredict').getOrCreate()


if __name__ == "__main__":
    arg = sys.argv[1:]
    if len(arg) != 2:
        sys.exit("Input and Target path are require.")
    else:
        main(arg)

