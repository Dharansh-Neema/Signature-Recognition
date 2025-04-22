from dataclasses import dataclass

@dataclass
class DataTransformationArtifact:
    train_data: str
    valid_data: str
    test_data: str
    classes:int

    def to_dict(self):
        return self.__dict__
