
import json
import random
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import Subset
from sklearn.model_selection import train_test_split
from data_ingestion import DataIngestion

class Split:

    def __init__(self, data_ingestor: DataIngestion, batch_size: int):
        '''
        Split the main dataset final_db.json into a train and a test set, which will
        be used for analysis and validation of the model
        '''
        self.my_db_class = data_ingestor
        self.train_loader = None
        self.test_loader = None
        self.batch_size = batch_size

    def train_test(self):
        dataset = self.get_dataset(self.data_storer)
        train_index, test_index = train_test_split(range(len(dataset)), test_size=0.3)

        train_set = Subset(dataset, train_index)
        self.train_loader = DataLoader(train_dataset, self.batch_size, shuffle=True)
        test_set = Subset(dataset, test_index)
        self.test_loader = DataLoader(test_dataset, self.batch_size, shuffle=False)
        
    # evaluation
    def eval_net(self, net, test_loader,epoch_i):
      net = net.eval()

      acc = 0
      n_pc = 0

      for batch_it, (x_tr,y_tr) in enumerate(self.test_loader):
        x_input = torch.FloatTensor(x_tr).cuda()
        y_input = torch.LongTensor(y_tr).cuda()

        probs = net(x_input)

        acc += (probs.data.cpu().argmax(dim = 1) == y_tr).sum()
        n_pc += y_tr.shape[0]

        acc = float(acc.data.cpu().numpy()) / n_pc

      print("test accuracy: %.2f" % acc)


    #training
    def train_net(net, n_epochs, train_loader, test_loader):
      loss_fcn = torch.nn.CrossEntropyLoss()
      minimizer = torch.optim.Adam(net.parameters(), lr = 0.0001)

      losses = []

      for epoch_i in range(10):
        print("EPOCH %s" % epoch_i)
        with torch.no_grad():
          eval_net(net, test_loader, epoch_i)
        net = net.train()

        acc = 0
        n_pc = 0

        for batch_it, (x_tr,y_tr) in enumerate[train_loader]:
          x_input = torch.FloatTensor(x_tr).cuda()
          y_input = torch.LongTensor(y_tr).cuda()

          probs = net(x_input)

          % print(y_input, probs)
          loss = loss_fcn(probs, y_input).mean()

          minimizer.zero_grad()
          loss.backward()
          minimizer.step()

          acc += (probs.data.cpu().argmax(dim = 1) == y_tr).sum()
          n_pc += y_tr.shape[0]

          losses.append(loss.item())

        ipython_display.display(pl.gcf)
        ipython_display.clear_output(wait = True)
        plt.title("Loss")
        plt.plot(losses)
        plt.show()





