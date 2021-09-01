from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from sklearn.model_selection import train_test_split
from data_ingestion import DataIngestion
import csv

class Split:

    def __init__(self, data_ingestor: DataIngestion, batch_size: int):
        '''
        Split the main dataset final_db.json into a train and a test set, which will
        be used for analysis and validation of the model
        '''
        self.my_db_class = data_ingestor
        self.batch_size = batch_size
        self.train_loader = None
        self.test_loader = None

    def train_test(self):
        dataset = []
        with open("final_images.csv", mode="r") as f:
            csv_reader = csv.DictReader(f)
            for row in csv_reader:
                dataset.append([row["lego_name"], row['point_cloud']])

        # 70/30 validation set approach with random state for reproducible output across multiple calls
        train_index, test_index = train_test_split(range(len(dataset)), test_size=0.3, random_state=2)
        train_set = Subset(dataset, train_index)
        self.train_loader = DataLoader(train_set, self.batch_size, shuffle=True)
        test_set = Subset(dataset, test_index)
        self.test_loader = DataLoader(test_set, self.batch_size, shuffle=False)


