import statistics
import torch.optim as optim
import wandb
from model import *
import open3d as o3d

def count_parameters(model):
    """
    Counts the number of parameters embedded in the model

    @param model: initialise a specific model
    @return int: the sum of parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class PointNetClassifier:

    def __init__(self, n_epochs: int, learning_rate, feature_transform=False):
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



            # POINTCLOUD REPRESENTATION
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(train_x[0].T)

            # outlier removal
            voxel_down_pcd = pcd.voxel_down_sample(voxel_size=0.02)
            cl, ind = voxel_down_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

            def display_inlier_outlier(cloud, ind):
                inlier_cloud = cloud.select_down_sample(ind)
                outlier_cloud = cloud.select_down_sample(ind, invert=True)
                # showing outliers
                outlier_cloud.paint_uniform_color([1, 0, 0])
                inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
                o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

            display_inlier_outlier(voxel_down_pcd, ind)

            # pcd.paint_uniform_color([0, 0, 0])
            # o3d.visualization.draw_geometries([pcd])

            loss = F.cross_entropy(preds, train_y)
            if trans is not None:
                loss_reg = feature_transform_regularizer(trans)
                loss += 0.5 * loss_reg

            loss.backward()

            optimizer.step()
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

            scheduler.step()
            model.eval()

    def test_net(self, test_loader, model):
        def test_step(test_x, test_y):
            test_x = torch.transpose(test_x, 2, 1)
            with torch.no_grad():
                preds, trans = model(test_x.float())
            loss = F.cross_entropy(preds, test_y)
            return loss, preds

        for i, batch in enumerate(test_loader):
            data, labels = batch
            batch_loss, preds = test_step(data, labels)

            batch_preds = torch.max(preds, -1).indices
            test_preds.append(batch_preds)
            test_labels.append(labels)

            test_preds = torch.cat(test_preds, dim=0).view(-1)
            test_labels = torch.cat(test_labels, dim=0).view(-1)

            assert test_preds.shape[0] == test_labels.shape[0]

            correct_classifications = torch.eq(test_labels, test_preds).sum()
            test_accuracy = (correct_classifications / test_labels.shape[0]) * 100
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy

        print("Final accuracy:{}".format(best_accuracy))
