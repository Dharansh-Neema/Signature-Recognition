import sys
from src.logger import logging
from src.entity.config_entity import DataTransformationConfig
from src.components.data_transformation import DataTransformation
from src.entity.artificat_entity import DataTransformationArtifact
from src.exception import CustomException

class TrainingPipeline:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

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
    
    def run_pipeline(self)->None:
        try:
            logging.info("Entered the run_pipeline method of TrainPipeline class")
            data_transformation_artificats = self.start_data_transformation()
            logging.info("Completed the run_pipeline method of TrainPipeline class")
        except Exception as e:
            raise CustomException(e, sys) from e