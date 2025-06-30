from src.cnnClassifier.config.configuration import ConfigurationManager
from src.cnnClassifier.components.model_evaluation import Evaluation

class ModelEvaluationTrainingPipeline:
    def __init__(self):
        pass
    
    def initiate_model_evaluation(self):
        try:
            config = ConfigurationManager()
            val_config = config.get_validation_config()
            evaluation = Evaluation(val_config)
            evaluation.evaluation()
            evaluation.save_score()

        except Exception as e:
            raise e