import json

import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from data_ingestion import DataIngestion
import csv
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch

class Split:

    def __init__(self, data_ingestor: DataIngestion, batch_size: int):
        '''
        Split the main dataset final_db.json into a train and a test set, which will
        be used for analysis and validation of the net_model
        '''
        self.my_db_class = data_ingestor
        self.batch_size = batch_size
        self.train_loader = None
        self.test_loader = None

    def train_test(self):
        dataset = []
        with open("final_images.csv", mode="r") as f:
            csv_reader = csv.DictReader(f)
            for row in csv_reader:
                x = row['point_cloud'].replace("["," ").replace("]", " ")

                ### QUIIIII ####
                x = np.fromstring(x, sep = " ").reshape((- 1, 3))

                dataset.append([row["lego_name"], x])

        # 70/30 validation set approach with random state for reproducible output across multiple calls

        train_index, test_index = train_test_split(range(len(dataset)), test_size=0.3, random_state=2)

        train_x = []
        train_y = []
        test_x = []
        test_y = []

        for index in range(len(dataset)):
            if index in train_index:
                train_x.append(dataset[index][1])
                train_y.append(dataset[index][0])
            elif index in test_index:
                test_x.append(dataset[index][1])
                test_y.append(dataset[index][0])
            else:
                break


        le = preprocessing.LabelEncoder()
        train_y = le.fit_transform(train_y)
        test_y = le.fit_transform(test_y)
        train_y = torch.as_tensor(train_y)
        test_y = torch.as_tensor(test_y)



        # transform to torch tensor
        #print(train_x[43])
        #train_x = torch.stack([torch.Tensor(el) for el in train_x])
        #test_x = torch.stack([torch.Tensor(el) for el in test_x])
        #train_y = torch.Tensor(train_y)
        test_x = torch.Tensor(test_x)
        #test_y = torch.Tensor(test_y)
        print(type(test_x))
        train_set = Dataset(train_x, train_y)
        test_set = Dataset(test_x, test_y)
        print("hi there")
        self.train_loader = DataLoader(train_set, self.batch_size, shuffle=True, num_workers=2)
        self.test_loader = DataLoader(test_set, self.batch_size, shuffle=False, num_workers=2)

    def get_train(self):
        return print(self.train_loader)

    def get_test(self):
        return self.test_loader


