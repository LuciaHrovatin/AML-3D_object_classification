import torch
import torch.nn as nn
import torch.optim as optim



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class PointNetClassifier:
    def __init__(self, batch_size: int, n_epochs: int, learning_rate=0.0001):
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

    def train_net(self, train_loader, test_loader, net_model):
        loss_fnc = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(net_model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96)

        def train_step(data, labels):
            optimizer.zero_grad()
            data = data
            labels = labels

            preds = net_model(data)
            loss = loss_fnc(preds, labels)

            loss.backward()
            optimizer.step()

            correct_classification = torch.eq(labels, torch.max(preds, -1).indices)
            accuracy = torch.sum(correct_classification).float() / labels.shape[0] * 100
            return loss.cpu().detach(), accuracy.cpu()

        def test_step(data, labels):
            data = data
            labels = labels
            # we deactivate torch autograd
            with torch.no_grad():
                preds = net_model(data)
            loss = loss_fnc(preds, labels)
            return loss.cpu(), preds.cpu()

        best_accuracy = 0.0
        for e in range(self.n_epochs):
            # we activate dropout, BN params
            net_model.train()

            for i, batch in enumerate(train_loader):
                data, labels = batch
                batch_loss, batch_accuracy = train_step(data, labels)

                if i % 1 == 0:
                    print('[{0}-{1:03}] loss: {2:0.05}, batch_accuracy: {3:0.03}'.format(
                        e + 1, i,
                        batch_loss.numpy(),
                        batch_accuracy.numpy()))
                if i == 0:
                    print('number of net_model parameters {}'.format(count_parameters(net_model)))

            # we call scheduler to decrease LR
            scheduler.step()

            net_model.eval()

            # Testing the whole test dataset
            test_preds = []
            test_labels = []
            total_loss = list()
            for i, batch in enumerate(test_loader):
                data, labels = batch
                batch_loss, preds = test_step(data, labels)

                batch_preds = torch.max(preds, -1).indices
                test_preds.append(batch_preds)
                test_labels.append(labels)
                total_loss.append(batch_loss)

            test_preds = torch.cat(test_preds, dim=0).view(-1)
            test_labels = torch.cat(test_labels, dim=0).view(-1)

            assert test_preds.shape[0] == test_labels.shape[0]

            loss = sum(total_loss) / len(total_loss)
            eq = torch.eq(test_labels, test_preds).sum()
            test_accuracy = (eq / test_labels.shape[0]) * 100
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
            print('End of Epoch {0}/{1:03} -> loss: {2:0.05}, test accuracy: {3:0.03} - best accuracy: {4:0.03}'.format(
                e + 1, self.n_epochs,
                loss.numpy(),
                test_accuracy.numpy(),
                best_accuracy))



