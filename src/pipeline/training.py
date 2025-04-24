import sys
from src.logger import logging
from src.entity.config_entity import DataTransformationConfig,ModelTrainerConfig
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.entity.artificat_entity import DataTransformationArtifact,ModelTrainerArtifact
from src.exception import CustomException

class TrainingPipeline:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        self.model_trainer_config = ModelTrainerConfig()

    def start_data_transformation(self):
        try:
            logging.info("Entered the start_data_transformation method of TrainPipeline class")
            data_transformation = DataTransformation(
                data_transformation_config=self.data_transformation_config
            )
            data_transformation_artificats = data_transformation.initiate_data_transformatio()

            
            logging.info("Completed the start_data_transformation method of TrainPipeline class")
            return data_transformation_artificats
        except Exception as e:
            raise CustomException(e, sys) from e
    

    def start_model_trainer(self,data_transformation_artifact:DataTransformationArtifact):
        try:
            logging.info("Entered the start_model_trainer method of TrainPipeline class")
            model_trainer = ModelTrainer(
                model_trainer_config=self.model_trainer_config,
                data_transformation_artifact=data_transformation_artifact
            )

            model_trainer_artifact = model_trainer.initiate_model_trainer()
            logging.info("Completed the start_model_trainer method of TrainPipeline class")
            return model_trainer_artifact
        except Exception as e:
            raise CustomException(e, sys) from e

    def run_pipeline(self)->None:
        try:
            logging.info("Entered the run_pipeline method of TrainPipeline class")
            data_transformation_artificats = self.start_data_transformation()
            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact=data_transformation_artificats)
            logging.info("Completed the run_pipeline method of TrainPipeline class")
        except Exception as e:
            raise CustomException(e, sys) from e