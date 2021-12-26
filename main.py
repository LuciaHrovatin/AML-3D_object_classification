from wandb.wandb_torch import torch
import wandb
from torch.utils.data import DataLoader
from data_ingestion import DataIngestion
from data_splitter import Split
from model import PointNetClassification
from solver import PointNetClassifier
import argparse


def main():
    arg_parser = argparse.ArgumentParser(description="AML 2021 - 3D object classification")
    arg_parser.add_argument("-p", "--points", required=False, default=1024, type=int,
                            help="Number of points per Point Cloud")
    arg_parser.add_argument("-s", "--db_sample", required=False, default=0, type=int,
                            help="Randomly sample the original dataset. The default model consider all the cases, by setting the parameter to 0.")
    arg_parser.add_argument("-t", "--train_test_split", required=False, default=0.3, type=float,
                            help="Train/test split size in percentage. Input the test size.")
    arg_parser.add_argument("-b", "--buffer", required=False, default=32, type=int, help="Buffer size")
    arg_parser.add_argument("-e", "--epochs", required=False, default=100, type=int, help="Epochs")
    arg_parser.add_argument("-lr", "--learning_rate", required=False, default=0.001, type=float, help="Learning rate")
    arg_parser.add_argument("-ft", "--feature_transform", required=False, default=True, type=bool, help="Feature transformation")
    args = arg_parser.parse_args()

    # ------- DATA INGESTION -------
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

    # Personal key to visualise the process on wandb
    wandb.login(key="5efd59f8e908e1fcc4a11a5654d956330bac1e0b")

    # ------- DATA SPLIT -------
    # define the number of classes
    num_classes = model_data.num_classes()
    # Slit in train/test sets
    split = Split(n_points=args.points, test_size=args.train_test_split, sample=args.db_sample)
    split.train_test()

    # Define the data loaders
    train_loader = DataLoader(split.get_train(), args.buffer, shuffle=True)
    test_loader = DataLoader(split.get_test(), args.buffer, shuffle=False)

    # ------- MODEL -------
    # 1. Start a new run
    wandb.init(project='pointNet-test', entity='aml_2021')

    # initialize the model
    model = PointNetClassification(n_classes=num_classes, feature_transform=args.feature_transform)

    # initialize the image Classifier
    image_classifier = PointNetClassifier(args.epochs, args.learning_rate, args.feature_transform)

    # train and evaluation
    image_classifier.train_net(train_loader, model)

    # test evaluation
    image_classifier.test_net(test_loader, model)


if __name__ == '__main__':
    main()
