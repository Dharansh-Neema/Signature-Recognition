import os 
import sys
import torch
from src.logger import logging
from src.exception import CustomException
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artificat_entity import ModelTrainerArtifact,DataTransformationArtifact
from src.utils.main_utils import save_object
from src.utils.main_utils import load_object
from tqdm import tqdm
import torch.nn as nn
from torchvision import models
from src.constants import DEVICE
from torch.utils.data import DataLoader


class ModelTrainer:
    def __init__(self,model_trainer_config:ModelTrainerConfig,data_transformation_artifact:DataTransformationArtifact) -> None:
        self.model_trainer_config = model_trainer_config
        self.data_transformation_artifact = data_transformation_artifact
        self.learning_rate = self.model_trainer_config.LR
        self.epochs = self.model_trainer_config.EPOCHS
        self.batch_size = self.model_trainer_config.BATCH_SIZE
        self.num_workers = self.model_trainer_config.NUM_WORKERS
        self.model_trainer_artifact = self.model_trainer_config.MODEL_TRAINER_ARTIFACTS_DIR
        self.trained_model_file_path = self.model_trainer_config.TRAINED_MODEL_FILE_PATH
        
    def train(self,model,criterion,optimizer,train_loader,valid_loader):
        """
        Method to train the model.
        Args:
            model (torch.nn.Module): The model to be trained.
            criterion (torch.nn.Module): The loss function.
            optimizer (torch.optim.Optimizer): The optimizer.
            train_loader (torch.utils.data.DataLoader): The training data loader.
            valid_loader (torch.utils.data.DataLoader): The validation data loader.
        """
        try:
            total_train_loss = 0
            total_test_loss = 0

            model.train()
            with tqdm(train_loader,unit='batch',leave=False) as pbar:
                pbar.set_description(f"Training")
                for images,idxs in pbar:
                    images = images.to(DEVICE,non_blocking=True)
                    idxs = idxs.to(DEVICE,non_blocking=True)
                
                    outputs = model(images)
                    loss = criterion(outputs, idxs)
                    
                    total_train_loss += loss.item()
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            model.eval()

            with tqdm(valid_loader,unit='batch',leave=False) as pbar:
                pbar.set_description(f"Testing")
                for images,idxs in pbar:
                    images = images.to(DEVICE,non_blocking=True)
                    idxs = idxs.to(DEVICE,non_blocking=True)
                    
                    outputs = model(images)
                    loss = criterion(outputs, idxs)
                    total_test_loss += loss.item()

            train_loss = total_train_loss / len(self.data_transformation_artifact.train_data)
            test_loss = total_test_loss / len(self.data_transformation_artifact.valid_data)

            print(f"Train Loss: {train_loss}")
            print(f"Test Loss: {test_loss}")

            # save_object(self.trained_model_file_path,model)

            # return train_loss,test_loss

        except Exception as e:
            raise CustomException(e, sys)
    

    def initiate_model_trainer(self)->ModelTrainerArtifact:
        """
        Method to initiate model trainer.
        Returns:
            ModelTrainerArtifact: The artifact of the model trainer.
        """
        try: 
            logging.info("Entered the initiate_model_trainer method of model trainer class")

            train_dataset = load_object(self.data_transformation_artifact.train_data)
            valid_dataset = load_object(self.data_transformation_artifact.valid_data)

            logging.info("Loading datasets completed")

            train_loader = DataLoader(train_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            shuffle=True)

            valid_loader = DataLoader(valid_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            shuffle=False)

            logging.info("Loading datasets completed")

            model = models.resnet34(weights="ResNet34_Weights.DEFAULT")
            logging.info("Loaded pre-trained model")

            model.fc = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(model.fc.in_features,self.data_transformation_artifact.classes)
            )
            logging.info("Model architecture Defined")

            model = model.to(DEVICE)

            criterion = torch.nn.CrossEntropyLoss()
            logging.info("Loss function defined")

            optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate,momentum=0.9)
            logging.info("Optimizer defined")
            logging.info("Starting model training")

            for i in range(self.epochs):
                logging.info(f"Epoch {i+1} of {self.epochs}")
                print(f"Epoch {i+1} of {self.epochs}")
                self.train(model,criterion,optimizer,train_loader,valid_loader)
            
            logging.info("Model Training completed!.")
            os.makedirs(self.model_trainer_config.MODEL_TRAINER_ARTIFACTS_DIR, exist_ok=True)
            torch.save(model,self.model_trainer_config.TRAINED_MODEL_FILE_PATH)
            logging.info(f"Model saved successfully at {self.model_trainer_config.TRAINED_MODEL_FILE_PATH}")
            model_trainer_artifacts = ModelTrainerArtifact(
                trained_model_path=self.model_trainer_config.TRAINED_MODEL_FILE_PATH)
            return model_trainer_artifacts
        except Exception as e:
            raise CustomException(e, sys)

        