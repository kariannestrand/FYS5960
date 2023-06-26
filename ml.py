import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedShuffleSplit
from matplotlib.colors import ListedColormap

class ML:

    ''' Class containing methods for performing training on dbs data using ml '''

    def __init__(self, read_filename, spare_size, epochs):
        self.epochs = epochs 

        df = pd.read_csv(read_filename + ".csv", sep = "\s+")
        self.num_parameters = len(df.columns) - 1
        self.stim_amp = df["stim_amp"].values
        self.min_dist = df["min_dist"].values
        self.spike = df["spike"].values

        X = df.iloc[:, 0:self.num_parameters].values  # array containing arrays of input values
        y = df.iloc[:, self.num_parameters].values    # array containing output values
        X[:, 0] = np.log10(- self.stim_amp)           # log10 of stimulus amplitude are used as parameter instead of stimulus amplitude
        self.normalize(X)                             # normalize the input arrays

        print(" ")
        print("Number of points along the axon: %d" % ((self.num_parameters - 3) / 3))

        count_neg = 0
        count_pos = 0 
        for i in range(len(y)):
            if (y[i] == 1):
                count_pos += 1
            else:
                count_neg += 1

        print(" ")
        print("Total number of negatives: %d" % count_neg)
        print("Total number of positives: %d" % count_pos)
        print(" ")
        self.count_neg = count_neg
        self.count_pos = count_pos

        # The StratifiedShuffleSplit class splits the data into training and spare sets while ensuring 
        # that each class is represented proportionally in both sets. 
        sss = StratifiedShuffleSplit(n_splits = 1, test_size = spare_size, random_state = 42)
        train_index, spare_index = next(sss.split(X, y))
        X_train, X_spare = X[train_index], X[spare_index]
        y_train, y_spare = y[train_index], y[spare_index]


        # splits the data into validation and test sets while ensuring 
        # that each class is represented proportionally in both sets.
        sss = StratifiedShuffleSplit(n_splits = 1, test_size = 0.5, random_state = 42)
        val_index, test_index = next(sss.split(X_spare, y_spare))
        X_val, X_test = X[val_index], X[test_index]
        y_val, y_test = y[val_index], y[test_index]


        self.X_train = torch.FloatTensor(X_train)
        self.X_val = torch.FloatTensor(X_val)
        self.X_test = torch.FloatTensor(X_test)
        self.y_train = torch.LongTensor(y_train)
        self.y_val = torch.LongTensor(y_val)
        self.y_test = torch.LongTensor(y_test)

    def scatter_plot(self, save_scatter_plot, name_scatter_plot):
        """ Method that creates a scatter plot to demonstrate how outcome is distinguished by 
            stimulus amplitude and minimum distance """
        stim_amp = self.stim_amp
        stim_amp_log = np.log10(- stim_amp)
        min_dist = self.min_dist
        spike = self.spike
        labels = ["No spike", "Spike"]
        cmap = "viridis"

        normalizing = lambda x: (x - np.min(x))/(np.max(x) - np.min(x))
        stim_amp_log_norm = normalizing(stim_amp_log)
        min_dist_norm = normalizing(min_dist)
    
        for i in range(len(stim_amp_log_norm)):
            if (stim_amp_log_norm[i] > 1 or stim_amp_log_norm[i] < 0):
                print("Normalization failed, stim_amp_log_norm[%d] = %.2f " % (i, stim_amp_log_norm[i]))
            elif (min_dist_norm[i] > 1 or min_dist_norm[i] < 0):
                print("Normalization failed, min_dist_norm[%d] = %.2f " % (i, min_dist_norm[i]))

        plt.figure(figsize = (7, 5))
        plt.scatter(stim_amp_log_norm, min_dist_norm, cmap = cmap, c = spike, s = 5)
        colorbar = plt.colorbar(ticks = [0, 1], format = plt.FuncFormatter(lambda i, *args: labels[int(i)]))
        colorbar.ax.tick_params(labelsize = 14)
        plt.xlabel("logarithm of stimulus amplitude (normalized)", fontsize = 14)#/[nA]", fontsize = 14)
        plt.ylabel("minimum distance (normalized)", fontsize = 14)#/[µm]", fontsize = 14)
        plt.xticks(fontsize = 12)
        plt.yticks(fontsize = 12)
        plt.tight_layout()
        if (save_scatter_plot == True):
            plt.savefig(name_scatter_plot)
        plt.show()

    def normalize(self, X):
        """ Method that normalizes the input array """
        num_parameters = self.num_parameters 

        for i in range(num_parameters):
            X_max = np.max(X[:, i])
            X_min = np.min(X[:, i])
            X[:, i] = (X[:, i] - X_min)/(X_max - X_min)
            for j in range(len(X)):
                if (X[j, i] > 1 or X[j, i] < 0):
                    print("Normalization failed, X[%d, %d] = %.2f " % (j, i, X[j, i]))

    def loss(self, model, criterion, optimizer, save_loss_plot, name_loss_plot, early_stop_patience):
        """ Method that calculate losses """
        epochs = self.epochs                   
        X_train = self.X_train
        X_val = self.X_val
        y_train = self.y_train
        y_val = self.y_val

        train_losses = []
        val_losses = []
        best_val_loss = np.inf
        epochs_without_improvement = 0
        for i in range(epochs):
            # Train the model
            y_pred_train = model(X_train).T[0]
            loss_train = criterion(y_pred_train, y_train.float())
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

            # Evaluate the model on validation data
            y_pred_val = model(X_val).T[0]
            loss_val = criterion(y_pred_val, y_val.float())

            train_losses.append(loss_train.item())
            val_losses.append(loss_val.item())

            if i % 10 == 0:
                print(f"epoch: {i} -> train loss: {loss_train}, val loss: {loss_val}")

            if loss_val < best_val_loss:
                best_val_loss = loss_val
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= early_stop_patience:
                    print(f"Stopping early after {i} epochs due to no improvement in validation loss")
                    break
            
        #plt.set_cmap("viridis")
        plt.figure(figsize = (7, 5))
        plt.plot(train_losses, marker = "o", linestyle = "-", label = "Training loss", markersize = 2, c = "#453781FF")
        plt.plot(val_losses, marker = "o", linestyle = "-", label = "Validation loss", markersize = 2, c = "#20A387FF")
        plt.xlabel("Epochs", fontsize = 14)
        plt.ylabel("Loss", fontsize = 14)
        plt.xticks(fontsize = 12)
        plt.yticks(fontsize = 12)
        plt.legend(fontsize = 12)
        if (save_loss_plot == True):
            plt.savefig(name_loss_plot)
        plt.show()

    def test(self, model):
        X_test = self.X_test
        y_test = self.y_test
        
        y_scatter = np.zeros(len(y_test))
        with torch.no_grad():
            TP = 0
            TN = 0
            FP = 0
            FN = 0
            for i, X in enumerate(X_test):
                y_pred = torch.round(model(X))

                if (y_pred == 1 and y_test[i] == 1):    # true positives
                    y_scatter[i] = 0
                    TP += 1
                elif (y_pred == 0 and y_test[i] == 0):  # true negatives
                    y_scatter[i] = 0
                    TN += 1
                elif (y_pred == 1 and y_test[i] == 0):  # false positives
                    y_scatter[i] = 1
                    FP += 1
                elif (y_pred == 0 and y_test[i] == 1):  # false negatives
                    y_scatter[i] = 2
                    FN += 1
        self.y_scatter = y_scatter 

        Accuracy = (TP + TN) / (TP + TN + FP + FN)
        Precision = TP / (TP + FP)
        Recall = TP / (TP + FN)
        F1 = (2 * Precision * Recall) / (Precision + Recall)  

        print(" ") 
        print(f"Number of true positives (TP) is {TP}.")
        print(f"Number of true negatives (TN) is {TN}.")
        print(f"Number of false positives (FP) is {FP}.")
        print(f"Number of false negatives (FN) is {FN}.")
        print(" ")
        print(f"Accuracy is {Accuracy}")
        print(f"Precision is {Precision}")
        print(f"Recall is {Recall}")
        print(f"F1 score is {F1}")
        print(" ")

    def test_scatter_plot(self, save_test_scatter_plot, name_test_scatter_plot):
        X_test = self.X_test   
        stim_amp_log_norm = X_test[:, 0]
        min_dist_norm = X_test[:, 2]
        y_scatter = self.y_scatter
        labels = ["TP and TN", "FP", "FN"] 
        colors = ["lightgray", "#20A387FF", "#453781FF"]
        cmap = ListedColormap(colors)

        plt.figure(figsize = (7, 5))
        plt.scatter(stim_amp_log_norm, min_dist_norm, c = y_scatter, 
                    cmap = cmap, s = [5 if label == 0 else 10 for label in y_scatter])
        colorbar = plt.colorbar(ticks = [0, 1, 2], format = plt.FuncFormatter(lambda i, *args: labels[int(i)]))
        colorbar.ax.tick_params(labelsize = 14)
        plt.xlabel("logarithm of stimulus amplitude (normalized)", fontsize = 14)
        plt.ylabel("minimum distance (normalized)", fontsize = 14)
        plt.xticks(fontsize = 12)
        plt.yticks(fontsize = 12)
        plt.tight_layout()
        if (save_test_scatter_plot == True):
            plt.savefig(name_test_scatter_plot)
        plt.show()


class NeuralNet(nn.Module):

    def __init__(self, num_parameters, dropout_rate):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(num_parameters, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(dropout_rate)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x
