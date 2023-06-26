import torch
import torch.nn as nn
import ml

spare_size = 0.2                        # fraction of dataframe used for validation and testing
epochs = 200                            # amount of times the dataframe is used for training
learning_rate = 0.01     
early_stop_patience = 10                # the number of epochs of which the validation loss has not improved before training is stopped
dropout_rate = 0.1                      # the rate at which to drop out neurons in the neural network during training 


save_scatter_plot = False
save_loss_plot = False
save_test_scatter_plot = False
name_scatter_plot = "scatter_straight_1000N.png"
name_loss_plot = "loss_straight_1000N_2P.png"
name_test_scatter_plot = "test_scatter_straight_1000N_2P.png"


def run_machine_learning():

    name = input("What is the name of the dataframe to be read? (e. g. 'straight_10000N_2P' excluding file-type): ")
    name_read_dataframe = "Data/" + name

    run_ml = ml.ML(name_read_dataframe, spare_size, epochs)
    run_ml.scatter_plot(save_scatter_plot, name_scatter_plot)

    model = ml.NeuralNet(run_ml.num_parameters, dropout_rate)
    count_neg = run_ml.count_neg
    count_pos = run_ml.count_pos
    class_weights = torch.tensor([float(count_neg)/float(count_pos)]) 
    criterion = nn.BCELoss(weight = class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

    run_ml.loss(model, criterion, optimizer, save_loss_plot, name_loss_plot, early_stop_patience)

    run_ml.test(model)
    run_ml.test_scatter_plot(save_test_scatter_plot, name_test_scatter_plot)
run_machine_learning()