import numpy as np 
import matplotlib.pyplot as plt

class DataFrame:

    def __init__(self, cell, num_axons, stim_pos):
        self.cell = cell 
        self.num_axons = num_axons
        self.stim_pos = stim_pos
        self.num_segments = self.cell.totnsegs

    def spike(self):
        """ Method that returns 1 if a spike is detected in any compartment of the axon, returns 0 if not """  

        cell = self.cell
        num_axons = self.num_axons

        if (np.isnan(cell.vmem).any()):
            print("Error! Membrane potential is NaN")
        max_vmem = np.max(cell.vmem)  
        if (max_vmem >= 0):
            if (num_axons == 1):
                print("Spike generated.")
            return 1
        else:
            if (num_axons == 1):
                print("Spike not generated.")
            return 0

    def minimum_distance(self):
        """ Method that finds and returns the minimum distance from the stimulus point source to the cell """

        cell = self.cell
        stim_pos = self.stim_pos
        num_segments = self.num_segments

        distance = np.zeros(num_segments);
       
        # Fill distance array with distances from stimulus point source to each compartment:
        for i in range(num_segments):
            x_i = np.abs(cell.x[i].mean() - stim_pos[0])
            y_i = np.abs(cell.y[i].mean() - stim_pos[1])
            z_i = np.abs(cell.z[i].mean() - stim_pos[2])
            distance[i] = np.sqrt(x_i**2 + y_i**2 + z_i**2)

        min_distance = np.min(distance)
        return min_distance

    def write_file(self, i, df, stim_amp, stim_dur, num_parameter_points):
        cell = self.cell
        num_axons = self.num_axons
        num_segments = self.num_segments
        step_size = int(num_segments / (num_parameter_points - 1))
        min_dist = self.minimum_distance()

        if (i == 0):
            df.write("stim_amp stim_dur min_dist ")
            for j in range(num_parameter_points):
                df.write("x%d y%d z%d " % (j, j, j))
            df.write("spike \n")

        df.write("%.3f %.3f %.3f " % (stim_amp[i], stim_dur[i], min_dist))
        for j in range(num_parameter_points):
            if (j == num_parameter_points - 1):
                df.write("%.3f %.3f %.3f " % (cell.x[-1].mean(), 
                                              cell.y[-1].mean(), 
                                              cell.z[-1].mean()))
            else:
                df.write("%.3f %.3f %.3f " % (cell.x[j*step_size].mean(), 
                                            cell.y[j*step_size].mean(), 
                                            cell.z[j*step_size].mean()))
        df.write("%d \n" % self.spike())

        if (i == num_axons-1):
            df.close()


class Visualizer:

    def __init__(self, cell, num_axons, stim_pos):
        self.cell = cell
        self.stim_pos = stim_pos
        self.num_axons = num_axons
        self.num_segments = self.cell.totnsegs

    def plot_3D(self, i, ax, save_3D_plot, name_3D_plot):
        """ Method that generates a 3D plot showing the position of the stimulus and axons """
        cell = self.cell 
        stim_pos = self.stim_pos 
        num_axons = self.num_axons


        if (i == 0):
            ax.scatter(stim_pos[0], stim_pos[1], stim_pos[2], label = "Stimulus", c = "#FDE725FF")
            ax.plot(cell.x.mean(axis = 1), cell.y.mean(axis = 1), cell.z.mean(axis = 1), label = "Axons", c = "#39568CFF")
        else:
            ax.plot(cell.x.mean(axis = 1), cell.y.mean(axis = 1), cell.z.mean(axis = 1), c = "#39568CFF")
        if (i == num_axons-1):
            ax.set_xlabel("x/[µm]"); ax.set_ylabel("y/[µm]"); ax.set_zlabel("z/[µm]")
            ax.tick_params(axis = 'both', labelsize = 12)
            ax.xaxis.label.set_size(14)
            ax.yaxis.label.set_size(14)
            ax.zaxis.label.set_size(14)

            ax.ticklabel_format(axis = "both", style = "sci", scilimits = (-3, 3), useOffset = False)
            ax.xaxis.get_offset_text().set_fontsize(12)
            ax.yaxis.get_offset_text().set_fontsize(12)
            ax.zaxis.get_offset_text().set_fontsize(12)

            ax.axes.set_xlim3d(left = -5e3, right = 5e3) 
            ax.axes.set_ylim3d(bottom = -5e3, top = 5e3) 
            ax.axes.set_zlim3d(bottom = -5e3, top = 5e3)

            ax.legend(fontsize = 12)
            if (save_3D_plot == True):
                plt.savefig(name_3D_plot)
            plt.show()

    def plot_subplot(self, stim_amp, stim_dur, ext_pot, save_subplot, name_subplot):
        cell = self.cell
        stim_pos = self.stim_pos 
        num_segments = self.num_segments
        
        stop_time = 20          
        time_step_size = 2**(-5) 
        num_time_steps = int(stop_time / time_step_size + 1) 
        time_vec = np.arange(num_time_steps) * time_step_size
        stim_delay = 5

        t0_idx = np.argmin(np.abs(time_vec - stim_delay))
        t1_idx = np.argmin(np.abs(time_vec - stim_delay - stim_dur))
        stim_current = np.zeros(num_time_steps)
        stim_current[t0_idx:t1_idx] = stim_amp

        pot_on_cell = ext_pot
        v_cell_ext = np.zeros((num_segments, num_time_steps))
        v_cell_ext[:, :] = np.dot(pot_on_cell[None, :].T, stim_current[:, None].T)

        cell_plot_idxs = np.array([0, int(cell.totnsegs / 2), cell.totnsegs - 1])
        cell_idx_clrs = lambda n: plt.cm.viridis(n / len(cell_plot_idxs))

        fig = plt.figure(figsize = [12, 6])
        fig.subplots_adjust(wspace = 0.6, left = 0.1, right = 0.98)
        ax1 = fig.add_subplot(141, aspect = 1, xlabel = "x/[µm]", ylabel = "z/[µm]", 
                              title = "axon and stim", xlim = [-3000, 3000], ylim = [-3000, 3000])
        ax2 = fig.add_subplot(142, xlabel = "time/[ms]", ylabel = "[µA]", title = "current pulse")
        ax3 = fig.add_subplot(143, xlabel = "time/[ms]", ylabel = "[mV]", title = "extracellular potential")
        ax4 = fig.add_subplot(144, xlabel = "time/[ms]", ylabel = "[mV]", title = "membrane potential")

        ax1.set_title("axon and stim", fontsize = 15)
        ax2.set_title("current pulse", fontsize = 15)
        ax3.set_title("extracellular potential", fontsize = 15)
        ax4.set_title("membrane potential", fontsize = 15)

        ax1.set_xlabel("x/[µm]", fontsize = 14)
        ax1.set_ylabel("z/[µm]", fontsize = 14)
        ax2.set_xlabel("time/[ms]", fontsize = 14)
        ax2.set_ylabel("[µA]", fontsize = 14)
        ax3.set_xlabel("time/[ms]", fontsize = 14)
        ax3.set_ylabel("[mV]", fontsize = 14)
        ax4.set_xlabel("time/[ms]", fontsize = 14)
        ax4.set_ylabel("[mV]", fontsize = 14)

        # Modify the x and y axis tick label size
        ax1.tick_params(axis = "both", labelsize = 12)
        ax2.tick_params(axis = "both", labelsize = 12)
        ax3.tick_params(axis = "both", labelsize = 12)
        ax4.tick_params(axis = "both", labelsize = 12)

        ax1.plot(cell.x.T, cell.z.T, c = "k")
        ax1.plot(stim_pos[0], stim_pos[2], "y*")
        for n, idx in enumerate(cell_plot_idxs):
            ax1.plot(cell.x[idx].mean(), cell.z[idx].mean(), "o", c = cell_idx_clrs(n))
            ax3.plot(cell.tvec, v_cell_ext[idx, :], c = cell_idx_clrs(n))
            ax4.plot(cell.tvec, cell.vmem[idx, :], c = cell_idx_clrs(n))


        ax2.plot(time_vec, stim_current / 1000, c = "k")
        
        if (save_subplot == True):
            plt.savefig(name_subplot)
        plt.show()
