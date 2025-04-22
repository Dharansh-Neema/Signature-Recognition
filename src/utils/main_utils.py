import os
import sys
from src.logger import logging
from src.exception import CustomException
import dill
import base64
import yaml
def save_obejct(file_path:str,obj:object)->None:
    logging.debug("Entered save object function")
    try:    
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path,"wb") as f:
            dill.dump(obj,f)
    except Exception as e:
        raise CustomException(e,sys) from e
    logging.debug("save object function completed")

def load_object(file_path:str)->object:
    logging.debug("Entered load object function")
    try:
        with open(file_path,"rb") as f:
            obj = dill.load(f)
        logging.debug("load object function completed")
        return obj
    except Exception as e:
        raise CustomException(e,sys) from e
   

def image_to_base64(image):
    try:
        logging.info("Entered the image_to_base64 method of utils")
        with open(image,"rb") as f:
            my_string = base64.b64encode(f.read())
        logging.info("image_to_base64 method completed")
        return my_string
    except Exception as e:
        raise CustomException(e,sys) from e

def read_yaml_file(file_path:str)->dict:
    """
    Reads a YAML file and returns the contents as a dictionary.
    file_path : str
    Returns : dict
    """
    try:
        with open(file_path,"r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise CustomException(e,sys) from e