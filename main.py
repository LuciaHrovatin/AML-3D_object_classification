import wandb
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

    wandb.login(key= "5efd59f8e908e1fcc4a11a5654d956330bac1e0b")


    num_classes = model_data.num_classes()

    split = Split()

    # generate train/test set
    split.train_test()

    train_loader = split.get_train() # Ottengo l'intero training set

    train_loader = DataLoader(train_loader, 32, shuffle=True, num_workers=2) # Estrazione di 120 casi
    #test_loader = split.get_test()

    ################## MODEL ###################
    # 1. Start a new run
    wandb.init(project='pointNet-test', entity='aml_2021')

    # initialize the model
    model = PointNetClassification(n_classes=num_classes, feature_transform=True)

    # initialize the image Classifier
    image_classifier = PointNetClassifier(n_epochs=100)

    # train and evaluation

    image_classifier.train_net(train_loader, model)



if __name__ == '__main__':
    main()
