import kagglehub

def data_ingestion(path_to_save: str = "./dataset"):
    """
    This function will download cedardataset data from kaggle and store it at root directory with name dataset
    """
    # Download latest version
    path = kagglehub.dataset_download("shreelakshmigp/cedardataset")
    print("Path to dataset files:", path)

data_ingestion()