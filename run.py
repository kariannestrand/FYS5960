import numpy as np 
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import dbs
import generate
import ml


num_axons = 100                         # number of simulated axons 
num_segments = 200                      # number of segments or compartments that comprise the axon
start_pos = [0, 0, -1500]               # [µm], start point of axon (x, y, z) if num_axons = 1
end_pos = [0, 0, 1500]                  # [µm], end point of axon (x, y, z) if num_axons = 1
stim_pos = np.array([0, 0, 0])          # [µm], position of stimulus (x, y z)                   
coord = ([[0, 0, -1500], 
          [0, 0, 0], 
          [0, 0, 1500]])                # [µm], coordinates of axon if coord_axon = True

stim_amp_min = 4.0                      # [nA], log exponent of maximum stimulus amplitude, log exponent of stimulus amplitude if num_axons = 1
stim_amp_max = 7.0                      # [nA], log exponent of minimum stimulus amplitude
stim_dur_min = 0.5                      # [ms], minimum stimulus duration, stimulus duration if num_axons = 1
stim_dur_max = 2.0                      # [ms], maximum stimulus duration 

ext_conductivity = 0.3                  # [S/m], extracellular conductivity 
spare_size = 0.2                        # fraction of dataframe used for validation and testing
epochs = 200                            # amount of times the dataframe is used for training
learning_rate = 0.01           


num_points = np.array([2, 5, 10, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200])
name_dataframes = []
for i in range(len(num_points)):
    name_dataframes.append("Straight_100N_" + str(num_points[i]) + "P")


save_dataframe = False                  # creates new dataframe(s) if True
coord_axon = False                      # axons are shaped with chosen coordinates in the list "coord" if True
interpolated_axons = False              # axons are shaped with polynomal interpolation if True
num_interpolation_points = 5            # number of points used to interpolate the shape of the axon
noise = 1000                            # amount of noise in interpolation
save_3D_plot = False
save_subplot = False
name_3D_plot = "3d_straight_100N.png"
name_subplot = "subplot_straight.png"


read_dataframe = False                  # dataframe is used for training, validation and testing if True
early_stop_patience = 10                # the number of epochs of which the validation loss has not improved before training is stopped
dropout_rate = 0.1                      # the rate at which to drop out neurons in the neural network during training 
save_scatter_plot = False
save_loss_plot = False
save_test_scatter_plot = False
name_scatter_plot = "scatter_straight_100N.png"
name_loss_plot = "loss_straight_100N_2P.png"
name_test_scatter_plot = "test_scatter_straight_100N_2P.png"
name_read_dataframe = "straight_100N_2P"


if (read_dataframe == False):
    run_axon = dbs.Axon(num_segments)
    if (interpolated_axons == True):
        locs = run_axon.shape_interpolated_axon(start_pos, end_pos, num_interpolation_points, noise)
    elif (coord_axon == True):
        locs = run_axon.shape_coord_axon(coord)
    else:
        locs = run_axon.shape_straight_axon(start_pos, end_pos)
    cell = run_axon.return_axon(locs)
    run_simulate = dbs.Simulate(cell, ext_conductivity, stim_pos)
    stim_amp = run_simulate.stim_amp_array(stim_amp_min, stim_amp_max, num_axons)
    stim_dur = run_simulate.stim_dur_array(stim_dur_min, stim_dur_max, num_axons)
    axon_pos = run_simulate.axon_pos_array(start_pos, num_axons)
    axon_rot = run_simulate.axon_rot_array(num_axons)
    run_visualizer = generate.Visualizer(cell, num_axons, stim_pos)
    run_dataframe = generate.DataFrame(cell, num_axons, stim_pos)
    if (save_dataframe == True):
        df = []
        for i in range(len(num_points)):
            df.append(open(name_dataframes[i] + ".csv", "w"))
    ax = plt.axes(projection = "3d")
    for i in tqdm(range(num_axons)):
        run_simulate.run_simulation(i, stim_amp, stim_dur, axon_pos, axon_rot)
        if (save_dataframe == True):
            for j in range(len(num_points)):
                run_dataframe.write_file(i, df[j], stim_amp, stim_dur, num_points[j])
        run_visualizer.plot_3D(i, ax, save_3D_plot, name_3D_plot)
    if (num_axons == 1):
        spike = run_dataframe.spike()
        ext_pot = run_simulate.ext_pot(cell.x.mean(axis = 1), cell.y.mean(axis = 1), cell.z.mean(axis = 1))
        run_visualizer.plot_subplot(stim_amp, stim_dur, ext_pot, save_subplot, name_subplot)
else:
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





