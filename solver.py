import statistics
import torch.optim as optim
from tqdm import tqdm

from model import *
import matplotlib.pyplot as plt


def count_parameters(model):
    """
    Counts the number of parameters embedded in the model

    @param model: initialise a specific model
    @return int: the sum of parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class PointNetClassifier:
    
    def __init__(self, n_epochs: int, learning_rate=0.0001, feature_transform=False):
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.feature_transform = feature_transform

    def train_net(self, train_loader, model):

        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, betas=(0.9, 0.999))
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

        def train_step(train_x, train_y):

            optimizer.zero_grad()

            train_x = torch.transpose(train_x, 2, 1)
            preds = model(train_x.float())

            loss = F.nll_loss(preds.requires_grad_(True), train_y)
            loss.backward()

            optimizer.step()
            correct_classification = torch.eq(train_y, torch.max(preds, -1).indices)
            accuracy = torch.sum(correct_classification).float() / train_y.shape[0] * 100
            return loss, accuracy



        def test_step(test_x, test_y):
            test_x = torch.transpose(test_x, 2, 1)
            with torch.no_grad():
                preds = model(test_x.float())
            loss = F.nll_loss(preds, test_y)
            return loss, preds

        best_accuracy = 0.0

        # for the graph
        losses = []
        accuracy_tot = []
        val = []

        checkpoint = torch.save({
            'epoch': 0,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': 0.0,
        }, "model.pt")

        for e in range(self.n_epochs):
            model.train()

            # for the graph
            batch_loss_value = []
            batch_accuracy_value = []
            for i, batch in enumerate(train_loader):
                data, labels = batch
                batch_loss, batch_accuracy = train_step(data, labels)

                # for the graph
                batch_loss_value.append(batch_loss.item())
                batch_accuracy_value.append(batch_accuracy.item())

                if i % 1 == 0:
                    print('[{0}-{1:03}] loss: {2:0.05}, batch_accuracy: {3:0.03}'.format(
                        e + 1, i,
                        batch_loss.detach().numpy(),
                        batch_accuracy.detach().numpy()))
                if i == 0:
                    print('number of model parameters {}'.format(count_parameters(model)))

            loss_epoch = sum(batch_loss_value)/len(batch_loss_value)
            accuracy_epoch = sum(batch_accuracy_value)/len(batch_accuracy_value)

            print(str(e + 1) + 'loss:' + str(round(loss_epoch,3)) + ' batch_accuracy:' + str(round(accuracy_epoch,3)))
            losses.append(statistics.mean(batch_loss_value))
            accuracy_tot.append(statistics.mean(batch_accuracy_value))
            scheduler.step()
            model.eval()

            checkpoint = torch.load("model.pt")
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            checkpoint['epoch'] = e+1
            checkpoint['loss'] = loss_epoch

            # Testing the model
            test_preds = []
            test_labels = []
            total_loss = []

            # It prints a graph at the end of the entire procedure
            plt.figure(figsize=(10, 5))
            plt.title("Training Loss")
            plt.plot(losses, label="train")
            plt.xlabel("n_epochs")
            plt.ylabel("Loss")
            plt.legend()
            plt.show()

            plt.figure(figsize=(10, 5))
            plt.title("Accuracy")
            plt.plot(accuracy_tot, label="train")
            plt.xlabel("n_epochs")
            plt.ylabel("Loss")
            plt.legend()
            plt.show()


"""
    def test_net(self, test_loader, model):
        for i, batch in enumerate(test_loader):
            data, labels = batch
            batch_loss, preds = test_step(data, labels)

            # for the graph
            val_losses.append(batch_loss.item())

            batch_preds = torch.max(preds, -1).indices
            test_preds.append(batch_preds)
            test_labels.append(labels)
            total_loss.append(batch_loss)

                # for the graph
            val.append(statistics.mean(val_losses))

            test_preds = torch.cat(test_preds, dim=0).view(-1)
            test_labels = torch.cat(test_labels, dim=0).view(-1)

            assert test_preds.shape[0] == test_labels.shape[0]

            loss = sum(total_loss) / len(total_loss)
            correct_classifications = torch.eq(test_labels, test_preds).sum()
            test_accuracy = (correct_classifications / test_labels.shape[0]) * 100
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
            print('End of Epoch {0}/{1:03} -> loss: {2:0.05}, test accuracy: {3:0.03} - best accuracy: {4:0.03}'.format(
                e + 1, self.n_epochs,
                loss.numpy(),
                test_accuracy.numpy(),
                best_accuracy))

            # It prints a graph at the end of the entire procedure
            plt.figure(figsize=(10, 5))
            plt.title("Training and Validation Loss")
            plt.plot(val, label="val")
            plt.plot(losses, label="train")
            plt.xlabel("n_epochs")
            plt.ylabel("Loss")
            plt.legend()
            plt.show()

"""
