import json
import pickle

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
        self.train_labels = None
        self.test_labels = None
        self.avg_pts = 0

    def train_test(self):
        with open("labels_final.pkl", "rb") as f:
            labels = pickle.load(f)
        with open("images_final.pkl", "rb") as f:
            images = pickle.load(f)

        # 70/30 validation set approach with random state for reproducible output across multiple calls
        train_index, test_index = train_test_split(range(len(images)), test_size=0.3, random_state=2)

        train_x = []
        train_y = []
        test_x = []
        test_y = []

        for index in range(len(images)):
            if index in train_index:
                train_x.append(images[index])
                train_y.append(labels[index])
            elif index in test_index:
                test_x.append(images[index])
                test_y.append(labels[index])
            else:
                break
            self.avg_pts += len(images[index])
        return print(self.avg_pts/len(images))

"""
        self.train_loader = train_x
        self.train_labels = train_y
        self.test_loader = test_x
        self.test_labels = test_y
"""
        """
        le = preprocessing.LabelEncoder()
        train_y = le.fit_transform(train_y)
        test_y = le.fit_transform(test_y)
        train_y = torch.as_tensor(train_y)
        test_y = torch.as_tensor(test_y)

        # transform to torch tensor
        train_x = torch.Tensor(train_x)
        test_x = torch.Tensor(test_x)
        train_set = TensorDataset(train_x, train_y)
        test_set = TensorDataset(test_x, test_y)
        print("hi there")
        self.train_loader = DataLoader(train_set, self.batch_size, shuffle=True, num_workers=2)
        self.test_loader = DataLoader(test_set, self.batch_size, shuffle=False, num_workers=2)
"""
    def get_train(self):
        return self.train_loader, self.train_labels

    def get_test(self):
        return self.test_loader, self.test_labels



