from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
import torch
import numpy as np
import pandas as pd


def median(x):
    """
    Computes the median of the points in a 3d Euclidean Space 
    (n-dimensional vector). 
    @return: the median value  
    """
    m, n = x.shape
    middle = np.arange((m - 1) >> 1, (m >> 1) + 1)
    x = np.partition(x, middle, axis=0)
    return x[middle].mean(axis=0)

def remove_outliers(data, thresh=2.0):
    """
    Exploits the geometric median to identify outliers 
    in 3d Euclidean Space and exclude them. 
    @return: cleaned Point Cloud or array 
    """
    m = median(data)
    s = np.abs(data - m)
    return data[(s < median(s) * thresh).all(axis=1)]

class Split:

    def __init__(self, n_points: int, test_size: float, sample: int):
        self.train_loader = None
        self.test_loader = None
        self.n_points = n_points
        self.test_size = test_size
        self.n_sample = sample

    def train_test(self):
        """
        Splits the dataset stored in images_final.pkl in a training and a validation set with
        proportion 70/30. The objects are randomly sampled, but the resulting sets can be reproduced across
        multiple calls.
        """

        images = pd.read_pickle("images_final.pkl")  # list of pictures as tensors
        labels = pd.read_pickle("labels_final.pkl")  # list of labels

        full_df = {"labels": labels,
                   "images": images
                   }

        full_df = pd.DataFrame(full_df)

        if not self.n_sample:
            # full dataset 
            train_index, test_index = train_test_split(range(len(full_df["images"])),
                                                       test_size=self.test_size,
                                                       random_state=2)
        else:
            # not full dataset 
            sub = full_df.sample(n=self.n_sample, replace=False, random_state=50)
            train_index, test_index = train_test_split(range(len(sub['images'])), test_size=self.test_size,
                                                       random_state=2)  # With subset of 3000 images

        train_x = []
        train_y = []
        test_x = []
        test_y = []

        # mapping each label to an integer number between 0 to 29  
        mapping = {'10b': 0,
                   '21': 1,
                   '2291': 2,
                   '236a': 3,
                   '2420': 4,
                   '2454': 5,
                   '2456': 6,
                   '250': 7,
                   '28': 8,
                   '3011': 9,
                   '30180': 10,
                   '3027': 11,
                   '303': 12,
                   '3030': 13,
                   '30355': 14,
                   '3300': 15,
                   '3433': 16,
                   '3685': 17,
                   '3747': 18,
                   '4019': 19,
                   '4772': 20,
                   '4854': 21,
                   '6156': 22,
                   '6213': 23,
                   '6215': 24,
                   '6474': 25,
                   '65735': 26,
                   '712': 27,
                   '9359': 28,
                   '971': 29}

        labels = [mapping[ll] for ll in labels]

        for index in range(len(images)):
            # removing the outliers
            no_outliers = remove_outliers(images[index])

            # exclude all the point clouds whose numerosity is less than n_points
            if index in train_index and len(no_outliers) > self.n_points:
                train_x.append(no_outliers)
                train_y.append(labels[index])
            elif index in test_index and len(no_outliers) > self.n_points:
                test_x.append(no_outliers)
                test_y.append(labels[index])

        train_y = torch.tensor(train_y)
        test_y = torch.tensor(test_y)

        # Populated with n random points
        test_x = torch.stack(
            [torch.from_numpy(el[np.random.choice(len(el), self.n_points, replace=False)]) for el in test_x])
        train_x = torch.stack(
            [torch.from_numpy(el[np.random.choice(len(el), self.n_points, replace=False)]) for el in train_x])

        train_set = TensorDataset(train_x, train_y)
        test_set = TensorDataset(test_x, test_y)
        self.train_loader = train_set
        self.test_loader = test_set

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



