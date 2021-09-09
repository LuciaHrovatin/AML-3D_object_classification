import pickle
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import torch
import numpy as np

class Split:

    def __init__(self, batch_size: int):
        self.batch_size = batch_size
        self.train_loader = None
        self.test_loader = None

    def train_test(self):
        """
        Splits the dataset stored in images_final.pkl in a training and a validation set with
        proportion 70/30. The objects are randomly sampled, but the resulting sets can be reproduced across
        multiple calls.
        """
        with open("labels_final.pkl", "rb") as f:
            labels = pickle.load(f)
        with open("images_final.pkl", "rb") as f:
            images = pickle.load(f)

        # 70/30 validation set approach with random state to reproduce the output across multiple calls
        train_index, test_index = train_test_split(range(len(images)), test_size=0.3, random_state=2)

        train_x = []
        train_y = []
        test_x = []
        test_y = []

        mapping = {'10b': 0,
                   '21': 1,
                   '2291': 2,
                   '236a': 3,
                   '2420':4,
                   '2454': 5,
                   '2456':6,
                   '250':7,
                   '28':8,
                   '3011':9,
                   '30180':10,
                   '3027': 11,
                   '303':12,
                   '3030':13,
                   '30355':14,
                   '3300':15,
                   '3433':16,
                   '3685':17,
                   '3747':18,
                   '4019':19,
                   '4772':20,
                   '4854':21,
                   '6156':22,
                   '6213':23,
                   '6215':24,
                   '6474':25,
                   '65735':26,
                   '712':27 ,
                   '9359':28,
                   '971':29}

        labels = [mapping[ll] for ll in labels]

        for index in range(len(images)):
            # exclude all the point clouds whose numerosity is less than 1024
            if index in train_index and len(images[index]) >= 1024:
                train_x.append(images[index])
                train_y.append(labels[index])
            elif index in test_index and len(images[index]) >= 1024:
                test_x.append(images[index])
                test_y.append(labels[index])

        #print(np.unique(np.array(train_y)))
        #print(np.unique(np.array(labels)))

        #le = preprocessing.LabelEncoder()
        #train_y = le.fit_transform(train_y)
        #test_y = le.fit_transform(test_y)
        train_y = torch.tensor(train_y)
        test_y = torch.tensor(test_y)

        train_x = torch.stack([torch.from_numpy(el[:1024]) for el in train_x])
        test_x = torch.stack([torch.from_numpy(el[:1024]) for el in test_x])
        train_set = TensorDataset(train_x, train_y)
        test_set = TensorDataset(test_x, test_y)
        self.train_loader = DataLoader(train_set, self.batch_size, shuffle=True, num_workers=2)
        self.test_loader = DataLoader(test_set, self.batch_size, shuffle=False, num_workers=2)

    def get_train(self):
        """
        When called, it returns the train_loader previously stored.

        @return: DataLoader containing the train_set and the corresponding labels (i.e., ground truth).
        """
        return self.train_loader

    def get_test(self):
        """
        When called, it returns the test_loader previously stored.
        
        @return: DataLoader containing the test_set and the corresponding labels (i.e., ground truth).
        """
        return self.test_loader
