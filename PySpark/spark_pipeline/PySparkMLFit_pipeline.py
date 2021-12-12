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

# Используйте как путь куда сохранить модель
#MODEL_PATH = 'spark_ml_model'


def process(spark, train_data, test_data):
    """
    Функция принимает на вход путь к фалам train и test. Обучает три модели
    DecisionTreeRegressor, RandomForestRegression и GBTRegressor.
    Выбирает модель с наименьшим RMSE и сохраняет ее в папку 'spark_ml_model'
    """
    
    df_train = spark.read.parquet(train_data)
    df_test = spark.read.parquet(test_data)
    
    stage_1 = VectorAssembler(inputCols=df_train.columns[1:-1], outputCol='features')#исключам ad_id
    
    #DecisionTreeRegression
    dtr = DecisionTreeRegressor(labelCol='ctr', featuresCol='features', predictionCol='prediction')   
    paramGrid_dtr = ParamGridBuilder()    .addGrid(dtr.maxDepth, [2,3,4,5])    .build()
    evaluator_dtr = RegressionEvaluator(metricName='rmse',
                                labelCol='ctr', 
                                predictionCol='prediction')
    tvs_dtr = TrainValidationSplit(estimator=dtr,
                           estimatorParamMaps=paramGrid_dtr,
                           evaluator= evaluator_dtr,
                           trainRatio=0.8)
    stage_2_dtr = TrainValidationSplit(estimator=dtr,
                           estimatorParamMaps=paramGrid_dtr,
                           evaluator= evaluator_dtr,
                           trainRatio=0.8)
    pipeline_dtr = Pipeline(stages=[stage_1, stage_2_dtr])
    model_dtr = pipeline_dtr.fit(df_train)
    predictions_dtr = model_dtr.transform(df_test)
    rmse_dtr = evaluator_dtr.evaluate(predictions_dtr)
    
    #RandomForestRegressor
    rfr = RandomForestRegressor(labelCol='ctr', featuresCol='features', predictionCol='prediction')   
    paramGrid_rfr = ParamGridBuilder()    .addGrid(rfr.maxDepth, [2,3,4,5])    .build()
    evaluator_rfr = RegressionEvaluator(metricName='rmse',
                                labelCol='ctr', 
                                predictionCol='prediction')
    tvs_rfr = TrainValidationSplit(estimator=rfr,
                           estimatorParamMaps=paramGrid_rfr,
                           evaluator= evaluator_rfr,
                           trainRatio=0.8)
    stage_2_rfr = TrainValidationSplit(estimator=rfr,
                           estimatorParamMaps=paramGrid_rfr,
                           evaluator= evaluator_rfr,
                           trainRatio=0.8)
    pipeline_rfr = Pipeline(stages=[stage_1, stage_2_rfr])
    model_rfr = pipeline_rfr.fit(df_train)
    predictions_rfr = model_rfr.transform(df_test)
    rmse_rfr = evaluator_rfr.evaluate(predictions_rfr)
    
    #GBTRegressor
    gbtr = GBTRegressor(labelCol='ctr', featuresCol='features', predictionCol='prediction')   
    paramGrid_gbtr = ParamGridBuilder()    .addGrid(gbtr.maxDepth, [2,3,4,5])    .build()
    evaluator_gbtr = RegressionEvaluator(metricName='rmse',
                                labelCol='ctr', 
                                predictionCol='prediction')
    tvs_gbtr = TrainValidationSplit(estimator=gbtr,
                           estimatorParamMaps=paramGrid_gbtr,
                           evaluator= evaluator_gbtr,
                           trainRatio=0.8)
    stage_2_gbtr = TrainValidationSplit(estimator=gbtr,
                           estimatorParamMaps=paramGrid_gbtr,
                           evaluator= evaluator_gbtr,
                           trainRatio=0.8)
    pipeline_gbtr = Pipeline(stages=[stage_1, stage_2_gbtr])
    model_gbtr = pipeline_gbtr.fit(df_train)
    predictions_gbtr = model_gbtr.transform(df_test)
    rmse_gbtr = evaluator_gbtr.evaluate(predictions_gbtr)
    
    if (rmse_dtr<rmse_rfr) and (rmse_dtr<rmse_gbtr):
        rmse = rmse_dtr
        #model_name = 'DecisionTreeRegressor'
        model = model_dtr
    elif (rmse_rfr<rmse_dtr) and (rmse_rfr<rmse_gbtr):
        rmse = rmse_rfr
        #model_name = 'RandomForestRegressor'
        model = model_rfr
    elif (rmse_gbtr<rmse_dtr) and (rmse_gbtr<rmse_rfr):
        rmse = rmse_gbtr
        #model_name = 'GBTRegressor'
        model = model_gbtr
    
    model.save('spark_ml_model')
    
    
    return print('RMSE = {}'.format(rmse))
    


def main(argv):
    train_data = argv[0]
    print("Input path to train data: " + train_data)
    test_data = argv[1]
    print("Input path to test data: " + test_data)
    spark = _spark_session()
    process(spark, train_data, test_data)


def _spark_session():
    return SparkSession.builder.appName('PySparkMLFitJob').getOrCreate()


if __name__ == "__main__":
    arg = sys.argv[1:]
    if len(arg) != 2:
        sys.exit("Train and test data are require.")
    else:
        main(arg)

