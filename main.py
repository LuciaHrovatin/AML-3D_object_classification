from data_ingestion import DataIngestion
from data_ingestion import *
#from train_test import Split


model = DataIngestion()
#model.extract_objects("./dataset/examples/") # saving the lego blocks in another folder
model.transform_csv("./images_final")





#split = Split(model)
#split.train_test()
