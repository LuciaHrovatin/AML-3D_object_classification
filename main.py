from data_ingestion import DataIngestion
from train_test import Split


model = DataIngestion()
print(model.extract_objects("./dataset/examples/"))
split = Split(model)
split.train_test()
