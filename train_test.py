
import json
import random
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import Subset
from sklearn.model_selection import train_test_split
from data_ingestion import DataIngestion

class Split:

    def __init__(self, data_ingestor: DataIngestion):
        '''
        Split the main dataset final_db.json into a train and a test set, which will
        be used for analysis and validation of the model
        '''
        self.my_db_class = data_ingestor

    def train_test(self):
        dataset = self.get_dataset(self.data_storer)
        train_index, valid_index = train_test_split(range(len(dataset)), test_size=0.3)

        batch_size = 100
        train_dataset = Subset(dataset, train_index)
        train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
        valid_dataset = Subset(dataset, valid_index)
        valid_dataloader = DataLoader(valid_dataset, batch_size, shuffle=False)
        return [train_dataloader, valid_dataloader]



"""
    def split(self):
        '''
        splits the original dataset into a train and a test set,
        given a determined splitting threshold (the variable split_th)

        @param self: the original json database
        @return train_set, test_set: json files containing training and test dataset
        '''
        with open(self.filename, "r") as f:
            f = json.load(f)
            n = len(f)
            
            random.seed(42)
            random.shuffle(f)

            split_th = int(0.7*n)
            train = f[: split_th]
            test = f[split_th:]

            with open("train_set.json", "w") as train_set:
                train_set.dump(train)
            
            with open("test_set.json", "w") as test_set:
                test_set.dump(test)

myfile = Split("https://github.com/LuciaHrovatin/AML-3D_object_classification/blob/main/final_db.json")
myfile.split()

"""





