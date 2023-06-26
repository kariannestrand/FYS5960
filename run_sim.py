import numpy as np 
import matplotlib.pyplot as plt
from tqdm import tqdm
import dbs
import generate


num_segments = 200                      # number of segments or compartments that comprise the axon
ext_conductivity = 0.3                  # [S/m], extracellular conductivity       
num_interpolation_points = 5            # number of points used to interpolate the shape of the axon
noise = 1000                            # amount of noise in interpolation

start_pos = [50, 0, -1500]              # [µm], start point of axon (x, y, z) if num_axons = 1
end_pos = [50, 0, 1500]                 # [µm], end point of axon (x, y, z) if num_axons = 1
stim_pos = np.array([0, 0, 0])          # [µm], position of stimulus (x, y z)                   
coord = ([start_pos, end_pos])

stim_amp_min = 4.0                      # [nA], log exponent of minimum stimulus amplitude, log exponent of stimulus amplitude if num_axons = 1
stim_amp_max = 7.0                      # [nA], log exponent of maximum stimulus amplitude
stim_dur_min = 0.5                      # [ms], minimum stimulus duration, stimulus duration if num_axons = 1
stim_dur_max = 2.0                      # [ms], maximum stimulus duration 


save_3D_plot = False 
save_subplot = False 
name_3D_plot = "Figures/3d_straight_1000N.png"
name_subplot = "Figures/subplot_straight.png"


def run_simulations():
    num_axons = int(input("Number of simulated axons (int): "))
    axon_shape = input("Do you want to simulate curved or straight axons? (c/s): ")
    shape = "straight" if (axon_shape == "s") else "curved"
    if (num_axons != 1):
        save_dataframes = input("Do you want to create new dataframes? (y/n): ")

    num_points = np.array([2, 5, 10, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200])
    name_dataframes = []
    for i in range(len(num_points)):
        name_dataframes.append("Data/" + shape + "_" + str(num_axons) + "N_" + str(num_points[i]) + "P")

    run_axon = dbs.Axon(num_segments)
    if (axon_shape == "c"):
        locs = run_axon.shape_curved_axon(start_pos, end_pos, num_interpolation_points, noise)
    else:
        locs = run_axon.shape_straight_axon(coord)
    cell = run_axon.return_axon(locs)

    run_simulate = dbs.Simulate(cell, ext_conductivity, stim_pos)
    stim_amp = run_simulate.stim_amp_array(stim_amp_min, stim_amp_max, num_axons)
    stim_dur = run_simulate.stim_dur_array(stim_dur_min, stim_dur_max, num_axons)
    axon_pos = run_simulate.axon_pos_array(start_pos, num_axons)
    axon_rot = run_simulate.axon_rot_array(num_axons)

    run_visualizer = generate.Visualizer(cell, num_axons, stim_pos)
    run_dataframe = generate.DataFrame(cell, num_axons, stim_pos)

    if (num_axons != 1):
        if (save_dataframes == "y"):
            df = []
            for i in range(len(num_points)):
                df.append(open(name_dataframes[i] + ".csv", "w"))

    ax = plt.axes(projection = "3d")
    for i in tqdm(range(num_axons)):
        run_simulate.run_simulation(i, stim_amp, stim_dur, axon_pos, axon_rot)
        if (num_axons != 1):
            if (save_dataframes == "y"):
                for j in range(len(num_points)):
                    run_dataframe.write_file(i, df[j], stim_amp, stim_dur, num_points[j])
        run_visualizer.plot_3D(i, ax, save_3D_plot, name_3D_plot)

    if (num_axons == 1):
        spike = run_dataframe.spike()
        ext_pot = run_simulate.ext_pot(cell.x.mean(axis = 1), cell.y.mean(axis = 1), cell.z.mean(axis = 1))
        run_visualizer.plot_subplot(stim_amp, stim_dur, ext_pot, save_subplot, name_subplot)
run_simulations()