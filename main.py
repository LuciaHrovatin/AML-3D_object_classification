from torch.utils.data import DataLoader

from data_ingestion import DataIngestion
from data_splitter import Split
from model import PointNetClassification
from solver import PointNetClassifier
from torch import torch

def main():

    ################## DATA INGESTION ###################

    model_data = DataIngestion()

    # If "lego_dataset.zip" uncomment the following lines (1, 2, 4):
    # 1. Unfold the data and save them in a folder
    # model_data.unzip_file("lego_dataset.zip")

    # 2. Extract lego block frames from the RGB and depth scenes
    # model_data.extract_objects("./dataset/examples/")

    # If "images_final.zip" uncomment ONLY the lines 3 - 4:
    # 3. Unfold "images_final.zip" and save them in a new folder
    # model_data.unzip_file("images_final.zip")

    # 4. Transform data in point clouds and save them in binary files
    # model_data.transform_binary("./images_final")

    ################## DATA SPLIT ###################

    num_classes = model_data.num_classes()

    # it is possible to change the batch size
    split = Split(120)

    # generate train/test set
    split.train_test()

    train_loader = split.get_train()
    train_loader = DataLoader(train_loader, 120, shuffle=True, num_workers=2)
    #test_loader = split.get_test()

    ################## MODEL ###################

    # initialize the model
    model = PointNetClassification(n_classes=num_classes, feature_transform=False)

    # initialize the image Classifier
    image_classifier = PointNetClassifier(n_epochs=100)



    # train and evaluation

    image_classifier.train_net(train_loader, model)


if __name__ == '__main__':
    main()
