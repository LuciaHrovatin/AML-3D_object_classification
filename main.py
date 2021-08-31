from data_ingestion import DataIngestion
#from train_test import Split


model = DataIngestion()
model.extract_objects("./dataset/examples/") # saving the lego blocks in another folder
#model.point_cloud(image_col="./images_final/scene0-20_view=0.jpeg_2420.jpeg",
#                  image_depth="./images_final/scene0-20_view=0_2420_depth.jpeg")


#split = Split(model)
#split.train_test()
