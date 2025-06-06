from dataclasses import dataclass
from src.utils.main_utils import read_yaml_file
from src.constants import CONFIG_PATH, ARTIFACT_DIR, DATA_TRANSFORMATION_ARTIFACTS_DIR, DATA_TRANSFORMATION_TRAIN_FILE_NAME, DATA_TRANSFORMATION_VALID_FILE_NAME, DATA_TRANSFORMATION_TEST_FILE_NAME,MODEL_TRAINER_ARTIFACTS_DIR,TRAINED_MODEL_PATH
import os

class DataTransformationConfig:
    def __init__(self):
        self.config = read_yaml_file(CONFIG_PATH)
        self.STD:list = self.config["data_transformation_config"]["std"]
        self.MEAN:list = self.config["data_transformation_config"]["mean"]
        self.IMG_SIZE:int = self.config["data_transformation_config"]["img_size"]
        self.DEGREE_N:int = self.config["data_transformation_config"]["degree_n"]
        self.DEGREE_P:int = self.config["data_transformation_config"]["degree_p"]
        self.TRAIN_RATIO:float = self.config["data_transformation_config"]["train_ratio"]
        self.VALID_RATIO:float = self.config["data_transformation_config"]["valid_ratio"]

        self.DATA_TRANSFORMATION_ARTIFACTS_DIR: str = os.path.join(os.getcwd(),ARTIFACT_DIR, DATA_TRANSFORMATION_ARTIFACTS_DIR)
        self.TRAIN_TRANSFORM_OBJECT_FILE_PATH : str = os.path.join(self.DATA_TRANSFORMATION_ARTIFACTS_DIR, DATA_TRANSFORMATION_TRAIN_FILE_NAME)
        self.VALID_TRANSFORM_OBJECT_FILE_PATH : str = os.path.join(self.DATA_TRANSFORMATION_ARTIFACTS_DIR, DATA_TRANSFORMATION_VALID_FILE_NAME)
        self.TEST_TRANSFORM_OBJECT_FILE_PATH : str = os.path.join(self.DATA_TRANSFORMATION_ARTIFACTS_DIR, DATA_TRANSFORMATION_TEST_FILE_NAME)


@dataclass
class ModelTrainerConfig:

    def __init__(self) -> None:
        self.config = read_yaml_file(CONFIG_PATH)
        self.LR : float = self.config['model_trainer_config']['lr']
        self.EPOCHS : int = self.config['model_trainer_config']['epochs']
        self.NUM_WORKERS : int = self.config['model_trainer_config']['num_workers']
        self.BATCH_SIZE : int = self.config['model_trainer_config']['batch_size']
        self.MODEL_TRAINER_ARTIFACTS_DIR :str = os.path.join(os.getcwd(),ARTIFACT_DIR,MODEL_TRAINER_ARTIFACTS_DIR)
        self.TRAINED_MODEL_FILE_PATH : str = os.path.join(self.MODEL_TRAINER_ARTIFACTS_DIR,TRAINED_MODEL_PATH)