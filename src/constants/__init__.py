import torch
import datetime as datetime
import os

CONFIG_PATH = os.path.join(os.getcwd(),"config","config.yaml")
TIMESTAMP = datetime.datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
ARTIFACT_DIR = os.path.join("artifacts",TIMESTAMP)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
