import torch
import torch.nn as nn
import torch.nn.functional as F  # Convolution functions
import numpy as np

"""
T-Net adds invariance to rotations

Consider we have a point cloud x - matrix of size 2048x3. T-net is a fcn of x that returns a matrix of shape 3x3. Since all points are located in a 3D space, we can consider the output of T-net as a transformation matrix and can multiply it on x.

x <- x ° Tnet(x),

where ° stands for matrix multiplication.

    We interpret TNet(x) as a matrix of transformation (i.e. rotation or reflection matrix)
      that takes the point cloud x as input and returns a 3x3 matrix
    We learn the transformation through NN-training

TL;DR: for the input point cloud x we predict and apply its rotation matrix


"""


class TNet3(nn.Module):

    def _init__(self):
        """
        prediction of a transformation matrix
        """
        super().__init__()

        # multi layer perceptron
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)

        # max pooling ?
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)

        # Rectifier Linear Units
        self.relu = nn.ReLU()

        # batch normalization (before or after ReLU?)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        # batchsize = x.size()[0]
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32)).view(1, 9).repeat
        if x.is_cuda:
            iden = iden.cuda()

        x += iden
        x = x.view(-1, 3, 3)

        return x

class TNet64(nn.Module):
    def __init__(self, k=64):
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

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


'''
Now we are ready to define PointNet. Our input will have shape [batch_size, n_point, 3].
The structure of the network should be the following:

    1. Linear convolution for each point(torch.nn.Conv1d)
    2. Batch Norm for each point (64)
    3. ReLU for each point
    4. Linear convolution for each point(torch.nn.Conv1d)
    5. Batch Norm for each point (128)
    6. ReLU for each point
    7. Permutation-invariant operation (max, sum)
    8. Multi-Layer perceptron
    9. Output size: [batch_size, num_classes]

'''


class PointNetFeature(nn.Module):

    def __init__(self, global_feat: True, feature_transform: False):
        super().__init__()
        self.tnet = TNet3()
        
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        
        self.main = nn.Sequential(
            torch.nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            torch.nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024)
        )
        
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.ftnet = TNet64(k=64)


    def forward(self, x):

        number_points = x.size()[2]
        trans = self.tnet(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.ftnet(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2,1)
        else:
            trans_feat = None

        pointfeatures = x
        x = self.main(x)

        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, number_points)
            return torch.cat([x, pointfeatures], 1), trans, trans_feat

class PointNetClassification(nn.Module):
    def __init__(self, k=2, feature_transform=False):
        super(PointNetClassification, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetFeature(global_feat=True, feature_transform=feature_transform)
        
        self.main = nn.Sequential(
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace = True),
            
            nn.Linear(512, 256),
            nn.Dropout(p=0.2),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace = True),
            
            nn.Linear(256, k)
            
        )
        

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = self.main(x)
        return F.log_softmax(x, dim=1), trans, trans_feat
    
    
"""
if __name__ == '__main__':
    sim_data = Variable(torch.rand(32,3,2500))
    trans = TNet3()
    out = trans(sim_data)
    print('stn', out.size())
    print('loss', loss_function(out))

    sim_data_64d = Variable(torch.rand(32, 64, 2500))
    trans = TNet64(k=64)
    out = trans(sim_data_64d)
    print('stn64d', out.size())
    print('loss', loss_function(out))

    pointfeat = PointNetfeat(global_feat=True)
    out, _, _ = pointfeat(sim_data)
    print('global feat', out.size())

    pointfeat = PointNetfeat(global_feat=False)
    out, _, _ = pointfeat(sim_data)
    print('point feat', out.size())

    classifier = PointNetClassification(k = 5)
    out, _, _ = classifier(sim_data)
    print('class', out.size())
"""