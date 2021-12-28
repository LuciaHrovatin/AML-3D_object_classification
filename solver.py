import os
import statistics
import torch.optim as optim
import wandb
from model import *


def count_parameters(model):
    """
    Counts the number of parameters embedded in the model

    @param model: initialise a specific model
    @return int: the sum of parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class PointNetClassifier:

    def __init__(self, n_epochs: int, learning_rate, feature_transform):
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.feature_transform = feature_transform

    def train_net(self, train_loader, model):

        # 2. Save model inputs and hyperparameters
        config = wandb.config
        config.learning_rate = self.learning_rate

        # 3. Log gradients and model parameters
        wandb.watch(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, betas=(0.9, 0.999))
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

        def train_step(train_x, train_y):

            optimizer.zero_grad()

            train_x = torch.transpose(train_x, 2, 1)
            preds, trans = model(train_x.float())

            loss = F.cross_entropy(preds, train_y)
            if trans is not None:
                loss_reg = feature_transform_regularizer(trans)
                loss += 0.5 * loss_reg

            loss.backward()

            optimizer.step()
            scheduler.step() # spostato qui lo step dello scheduler

            correct_classification = torch.eq(train_y, torch.max(preds, -1).indices)
            accuracy = torch.sum(correct_classification).float() / train_y.shape[0] * 100

            return loss, accuracy

        # for the graph
        losses = []
        accuracy_tot = []

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
                wandb.log({"loss": batch_loss,
                           "accuracy": batch_accuracy})
                if i % 10 == 0:
                    print('[{0}-{1:03}] loss: {2:0.05}, batch_accuracy: {3:0.03}'.format(
                        e + 1, i,
                        batch_loss.detach().numpy(),
                        batch_accuracy.detach().numpy()))

                if e == 0 and i == 0:
                    print('number of model parameters {}'.format(count_parameters(model)))

            loss_epoch = sum(batch_loss_value) / len(batch_loss_value)
            accuracy_epoch = sum(batch_accuracy_value) / len(batch_accuracy_value)

            print(str(e + 1) + 'loss:' + str(round(loss_epoch, 3)) + ' batch_accuracy:' + str(round(accuracy_epoch, 3)))
            
            losses.append(statistics.mean(batch_loss_value))
            accuracy_tot.append(statistics.mean(batch_accuracy_value))

            #scheduler.step()
            #model.eval()

        return torch.save(model.state_dict(), os.path.join(os.getcwd(), 'model.pth'))

    def test_net(self, test_loader, model):

        def test_step(test_x, test_y):

            model.eval() # mancava nella fcn di test?

            test_x = torch.transpose(test_x, 2, 1)
            with torch.no_grad():
                preds, trans = model(test_x.float())
            loss = F.cross_entropy(preds, test_y)
            num_correct = sum(torch.argmax(preds, dim=1) == test_y)
            return loss, num_correct

        test_loss = []
        test_acc = 0.0
        for i, batch in enumerate(test_loader):

            data, labels = batch
            batch_loss, num_correct = test_step(data, labels)
            test_loss.append(batch_loss/len(test_loader))
            test_acc += num_correct/len(test_loader)

        
        '''#Ipotetico codice da aggiungere :) 
        wandb.init(config=args)

        best_accuracy = 0.0
        for epoch in range(1, args.epochs + 1):
            test_loss, test_accuracy = test()
            if (test_accuracy > best_accuracy):
                wandb.run.summary["best_accuracy"] = test_accuracy
                best_accuracy = test_accuracy'''
        
        print("Final accuracy:{}".format(test_acc))
        print("Final loss:{}".format(np.mean(test_loss)))


''' def get_class_weight(sample_per_class, n_classes = num_classes, power = 1):
    """ Finds the weighted importance of the sample on the total number of classes
    @param sample_per_class: number of samples of each class
    @param num_classes: total number of classes (default: num_classes = 10)
    @param power: exponent of np power (default: 1)
    
    @return float: weight of class sample """

    weight_sample = 1.0/ np.array( np.power(sample_per_class, power))
    weight_sample = weight_sample/ np.sum(weight_sample) * num_classes

    return weight_sample '''