from src.cnnClassifier.pipeline.data_ingestion_pipeline import (
    DataIngestionTrainingPippeline,
)
from src.cnnClassifier.pipeline.prep_base_model_pipeline import (
    PrepareBaseModelTrainingPipeline,
)
from src.cnnClassifier.pipeline.training_pipeline import ModelTrainingPipeline
from src.cnnClassifier.pipeline.model_evaluation_pipeline import ModelEvaluationTrainingPipeline
from src.cnnClassifier import logger



STAGE_NAME = "Data Ingestion Stage"


try:
    data_ingestion = DataIngestionTrainingPippeline()
    data_ingestion.initiate_data_ingestion()
    logger.info(f"Stage {STAGE_NAME} completed")
except Exception as e:
    raise logger.info(e)


STAGE_NAME = "Prepare base model"
try:
    logger.info(f"*******************")
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    prepare_base_model = PrepareBaseModelTrainingPipeline()
    prepare_base_model.initiate_base_model()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Training"
try: 
    logger.info(f"*******************")
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    model_trainer = ModelTrainingPipeline()
    model_trainer.initiate_model_training()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e
    

STAGE_NAME = "Evaluation stage"
try:
   logger.info(f"*******************")
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   model_evalution = ModelEvaluationTrainingPipeline()
   model_evalution.initiate_model_evaluation()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

except Exception as e:
        logger.exception(e)
        raise e