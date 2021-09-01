from data_ingestion import DataIngestion
from train_test import Split

#### DATA INGESTION ##
model_data = DataIngestion()

# Unfold the data and save them in a folder
# model_data.unzip_file("lego_dataset.zip")

# Extract lego block frames from the RGB and depth scenes
# model_data.extract_objects("./dataset/examples/")

# Save lego block names and point clouds in a csv file
# model_data.transform_csv("./images_final")


#### DATA TRANSFORMATION ###

split = Split(model_data, 100)

# generate train/test set
split.train_test()
