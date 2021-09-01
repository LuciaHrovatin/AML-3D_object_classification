from data_ingestion import DataIngestion
from data_splitter import Split
from model import PointNetClassification
from solver import PointNetClassifier



def main():
    ### DATA INGESTION ##
    model_data = DataIngestion()

    # Unfold the data and save them in a folder
    # model_data.unzip_file("lego_dataset.zip")

    # Extract lego block frames from the RGB and depth scenes
    # model_data.extract_objects("./dataset/examples/")

    # Save lego block names and point clouds in a csv file
    # model_data.transform_csv("./images_final")

    #### DATA TRANSFORMATION ###

    split = Split(model_data, 50)

    # generate train/test set
    split.train_test()

    # init the model
    model = PointNetClassification()

    # dataloaders -> they use multiprocessing for loading data
    train_loader = split.get_train()
    test_loader = split.get_test()
    print("HERE")
    # init Image Classifier
    image_classifier = PointNetClassifier(batch_size=32, n_epochs=100)
    print("HERE2")
    # train and eval with pytorch classes
    image_classifier.train_net(train_loader, test_loader, model)


if __name__ == '__main__':
    main()
