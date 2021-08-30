from data_ingestion import DataIngestion
#from train_test import Split


model = DataIngestion()
model.extract_objects("./dataset/examples/") # save the lego images in another folder
#split = Split(model)
#split.train_test()
