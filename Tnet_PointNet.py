import torch
import torch.nn as nn
import torch.nn.functional as F # Convolution functions
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

class TNet(nn.Module):
  """
  prediction of a transformation matrix
  """

  def _init__(self):
    
     super().__init__()

    # multi layer perceptron
     self.conv1 = torch.nn.Conv1d(3,64,1)            #####
     self.conv2 = torch.nn.Conv1d(64,128,1)          #####
     self.conv3 = torch.nn.Conv1d(128,1024,1)        #####

     # max pooling ?
     self.fc1 = nn.Linear(1024, 512)                 #####
     self.fc2 = nn.Linear(512, 256)                  #####   ALL OF
     self.fc3 = nn.Linear(256, 9)                    #####   THIS PART

    # Rectifier Linear
     self.relu = nn.ReLU()                           #####   IS USED FOR

    # batch normalization (before or after ReLU?)    #####   SEMANTIC
     self.bn1 = nn.BatchNorm1d(64)                   #####   SEGMENTATION
     self.bn2 = nn.BatchNorm1d(128)                  #####
     self.bn3 = nn.BatchNorm1d(1024)                 #####
     self.bn4 = nn.BatchNorm1d(512)                  #####
     self.bn5 = nn.BatchNorm1d(256)                  #####


  def forward(self, x):
     #batchsize = x.size()[0]
     x = x.transpose(2,1)
     x = F.relu(self.bn1(self.conv1(x))) # here used before
     x = F.relu(self.bn2(self.conv2(x)))
     x = F.relu(self.bn3(self.conv3(x)))
     x = torch.max(x,2,keepdim = True)[0]
     x = x.view(-1, 1024)

     x = F.relu(self.bn4(self.fc1(x)))
     x = F.relu(self.bn5(self.fc2(x)))
     x = self.fc3(x)

     iden = torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32)).view(1,9).repeat
     if x.is_cuda:
       iden = iden.cuda()
    
     x += iden
     x = x.view(-1,3,3)

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


class PointNet(nn.Module):

  def __init__(self, num_classes:int):

    super().__init__()
    self.tnet = TNet()

    self.main = nn.Sequential(
        torch.nn.Conv1d(3,64,1),
        nn.BatchNorm1d(64),
        nn.ReLU(inplace=True),

        torch.nn.Conv1d(64,128,1),
        nn.BatchNorm1d(128),
        nn.ReLU(inplace=True),

        torch.nn.Conv1d(128,1024,1),
        nn.BatchNorm1d(1024)
    )

    self.linear = nn.Linear(1024, num_classes)
    self.softmax = nn.SoftMax(1)

  def forward(self, x):
    # [batch, n_points, 3]
    trans = self.tnet(x)
    x = torch.bmm(x, trans)

    x = x.transpose(1,2)
    # [batch, 3, n_points]
    x = self.main(x)
    # shape [batch_size, 1024, n_points]
    x, _ = torch.max(x,2)

    x = self.linear(x)
    x = self.softmax(x)

    return x