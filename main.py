from src.cnnClassifier.pipeline.data_ingestion_pipeline import (
    DataIngestionTrainingPippeline,
)
from src.cnnClassifier import logger


STAGE_NAME = "Data Ingestion Stage"

if __name__ == "__main__":
    try:
        data_ingestion = DataIngestionTrainingPippeline()
        data_ingestion.initiate_data_ingestion()
        logger.info(f"Stage {STAGE_NAME} completed")
    except Exception as e:
        raise logger.info(e)
