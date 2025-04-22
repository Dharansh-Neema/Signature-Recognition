import os
import sys
from src.logger import logging
from src.exception import CustomException
from torchvision import datasets, transforms as T
from src.utils.main_utils import save_object
from src.entity.config_entity import DataTransformationConfig
from data import signatures
from src.entity.artificat_entity import DataTransformationArtifact
import torch
class DataTransformation:
    def __init__(self, data_transformation_config:DataTransformationConfig):
        self.config = data_transformation_config
        self.dataset = signatures
        self.std = self.config.STD
        self.mean = self.config.MEAN
        self.img_size = self.config.IMG_SIZE
        self.degree_n = self.config.DEGREE_N
        self.degree_p = self.config.DEGREE_P
        self.train_ratio = self.config.TRAIN_RATIO
        self.valid_ratio = self.config.VALID_RATIO

    def data_transformation(self):
        """
            return transform data
        """
        try:
            logging.info("Entered the get_transform_data of Data Transformation class")
            data_transform = T.Compose([
                T.Resize(size=(self.img_size, self.img_size)),
                T.ToTensor(),
                # normalise by 3 means 3 std's of image net, 3 channels
                T.Normalize(mean=self.mean, std=self.std),
                T.RandomRotation(degrees=(self.degree_n, self.degree_p))
            ])
            logging.info("get_transform_data of Data Transformation class completed")
            return data_transform
        
        except Exception as e:
            raise CustomException(e, sys) from e
    
    def split_data(self,dataset,total_count):
        """
        This function split data into train,valid and test
        """
        try:
            logging.info("Entered the split_data of Data Transformation class")
            train_count = int(self.train_ratio*total_count)
            valid_count = int(self.valid_ratio*total_count)
            test_count = total_count - train_count - valid_count
            train_data,valid_data,test_data = torch.utils.data.random_split(
                dataset,(train_count,valid_count,test_count)
            )
            logging.info("split_data of Data Transformation class completed")
            return train_data,valid_data,test_data
        except Exception as e:
            raise CustomException(e, sys) from e

    def initiate_data_transformatio(self)->DataTransformationArtifact:
        """
        This function initiate data transformation and return data transformation artifact
        """
        try:
            logging.info("Entered the initiate_data_transformatio of Data Transformation class")
            data_dir = r"E:\ProjectsAI\Signature-Recognition\data\signatures"
            
            dataset = datasets.ImageFolder(data_dir,transform=self.data_transformation())
            total_count = len(dataset)
            
            classes = len(os.listdir(data_dir))
            print("classes count:",classes)

            train_data,valid_data,test_data = self.split_data(dataset,total_count)
            logging.info("split data completed")

            save_object(
                file_path=self.config.TRAIN_TRANSFORM_OBJECT_FILE_PATH,
                obj=train_data
            )
            save_object(
                file_path=self.config.VALID_TRANSFORM_OBJECT_FILE_PATH,
                obj=valid_data
            )
            save_object(
                file_path=self.config.TEST_TRANSFORM_OBJECT_FILE_PATH,
                obj=test_data
            )
            logging.info("data transformation completed")

            data_transformation_artifact = DataTransformationArtifact(
                train_data=self.config.TRAIN_TRANSFORM_OBJECT_FILE_PATH,
                valid_data=self.config.VALID_TRANSFORM_OBJECT_FILE_PATH,
                test_data=self.config.TEST_TRANSFORM_OBJECT_FILE_PATH,
                classes=classes
            )
            logging.info("data transformation artifact completed")
            return data_transformation_artifact
        except Exception as e:
            raise CustomException(e, sys) from e