import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable


class TNet3(nn.Module):
    def __init__(self):
        """
        TNet3(x) is an input alignment network. It is a matrix of transformation
        (i.e., rotation or reflection matrix) that takes the Point Cloud x as input
        and returns a 3x3 matrix adding invariance to rotations.
        """
        super(TNet3, self).__init__()

        # multi layer perceptron
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)

        # Rectifier Linear Units
        self.relu = nn.ReLU()

        # batch normalization
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batch_size = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batch_size, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x += iden
        x = x.view(-1, 3, 3)

        return x


class TNet64(nn.Module):
    def __init__(self, k=64):
        """
        TNet64(x, n_classes) is a point feature alignment network. It resembles the big network
        and is composed by basic modules of point independent feature extraction, max pooling
        and fully connected layers.

        @param k: set to 64 accordingly to the original PointNet architecture
        """
        super(TNet64, self).__init__()

        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)

        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x += iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetFeature(nn.Module):
    def __init__(self, global_feature: True, feature_transform):
        """
        Internal network whose input has shape [batch_size, 1024, 3]. It is structured
        following the directives of the original PointNet architecture.

        @param global_feature: by default it is set to True
        @param feature_transform: by default it is set up to False. If switched to True,
        a point feature alignment involving the TNet64() is performed
        """
        super(PointNetFeature, self).__init__()
        self.tnet_input = TNet3()

        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.global_feat = global_feature
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.feature_net = TNet64()

    def forward(self, x):
        number_points = x.size()[2]
        transformed_matrix = self.tnet_input(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, transformed_matrix)
        x = x.transpose(2, 1)

        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            transformed_features = self.feature_net(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, transformed_features)
            x = x.transpose(2, 1)
        else:
            transformed_features = None

        point_features = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        # MAX pooling
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        if self.global_feat:
            return x, transformed_matrix, transformed_features
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, number_points)
            return torch.cat([x, point_features], 1), transformed_matrix, transformed_features


class PointNetClassification(nn.Module):
    def __init__(self, n_classes: int, feature_transform):
        """
        Last Multi-Layer Perceptron classifier trained  on the shape global features for block classification.

        @param n_classes: total number of classes (i.e., Lego pieces)
        @param feature_transform: False by default. If switched to True, a point feature
        alignment involving the TNet64() is performed.
        """
        super(PointNetClassification, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetFeature(global_feature=True, feature_transform=feature_transform)

        # Fully connected layers
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, n_classes)

        # dropout method with keep ratio 0.7 --> p=0.3
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)

        # BISOGNA considerarla oppure no?
        #x = F.log_softmax(x, dim=1)
        if self.feature_transform:
            return x, trans
        else:
            return x, None


def feature_transform_regularizer(transformed_matrix):
    """
    If the TNet64  is included in the architecture, a further regularization term
    to the softmax training loss is required. This procedure should constrain the
    points feature transformation matrix to be close to an orthogonal matrix.
    
    @param transformed_matrix: matrix coming from the TNet64 alignment network
    @return: regularized softmax loss
    """
    d = transformed_matrix.size()[1]
    Identity_matrix = torch.eye(d)[None, :, :]
    loss = torch.mean(torch.norm(torch.bmm(transformed_matrix, transformed_matrix.transpose(2, 1)) - Identity_matrix, dim=(1, 2)))
    return loss
