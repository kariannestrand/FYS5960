import numpy as np 
from scipy.interpolate import CubicSpline
import neuron
import LFPy
h = neuron.h

class Axon:

    def __init__(self, num_segments):
        self.num_segments = num_segments                                               # [ms]

    def shape_curved_axon(self, start_pos, end_pos, num_interpolation_points, noise): 
        num_segments = self.num_segments

        seg_len_frac = np.linspace(0, 1, num_interpolation_points)
        
        x_orig = np.linspace(start_pos[0], end_pos[0], num_interpolation_points)     
        y_orig = np.linspace(start_pos[1], end_pos[1], num_interpolation_points)    

        x_orig += np.random.normal(0, noise, size = num_interpolation_points)
        y_orig += np.random.normal(0, noise, size = num_interpolation_points)
        
        cs_x = CubicSpline(seg_len_frac, x_orig)  
        cs_y = CubicSpline(seg_len_frac, y_orig)

        seg_length_new = np.linspace(0, 1, num_segments)    

        x_new = cs_x(seg_length_new)
        y_new = cs_y(seg_length_new)
        z_new = np.linspace(start_pos[2], end_pos[2], num_segments)

        locs = np.zeros((num_segments, 4))
        locs[:, 0] = x_new
        locs[:, 1] = y_new
        locs[:, 2] = z_new
        locs[:, 3] = np.ones(num_segments)
        return locs

    def shape_straight_axon(self, coord):
        x = []
        y = []
        z = []

        for i in range(len(coord)):
            x.append(coord[i][0])
            y.append(coord[i][1])
            z.append(coord[i][2])
        
        locs = np.zeros((len(coord), 4))
        locs[:, 0] = x
        locs[:, 1] = y
        locs[:, 2] = z
        locs[:, 3] = np.ones(len(coord))
        return locs

    def return_axon(self, locs):
        num_segments = self.num_segments
        stop_time = 20
        time_step_size = 2**(-5)
        
        axon = h.Section(name = "axon")
        h.pt3dclear(sec = axon)
        axon.nseg = num_segments
        h.pt3dadd(h.Vector(locs[:, 0]), 
                  h.Vector(locs[:, 1]), 
                  h.Vector(locs[:, 2]), 
                  h.Vector(locs[:, 3]), sec = axon)
        h.define_shape()
        axon.insert("hh")
        axon.Ra = 150                     # [Ωcm], axial resistivity
        axon.cm = 1.                      # [μF/cm^2], membrane capacitance

        all_sec = h.SectionList()         # class for creating and managing a list of sections
        for sec in h.allsec():
            all_sec.append(sec = sec)
        cell_params = {
                "morphology": all_sec,    # str or neuron.h.SectionList
                "delete_sections": False, # delete pre-existing section-references, defaults to True
                "v_init": -65.,           # initial membrane potential 
                "passive": False,         # passive mechanisms are initialized if True, defaults to False
                "nsegs_method": None,     # "lambda100", "lambda_f", "fixed_length" or None, determines 
                                          # number of segments, defaults to 'lambda100'
                "dt": time_step_size,     # simulation timestep, defaults to 2^-4 ms
                "tstart": -100.,          # initialization time for simulation <= 0 ms, defaults to 0
                "tstop": stop_time,       # stop time for simulation > 0 ms, defaults to 100 ms
                "extracellular": True,    # switch for NEURON's extracellular mechanism, defaults to False
            }
        cell = LFPy.Cell(**cell_params)
        cell.__axon__ = axon
        return cell
    
    
class Simulate:
    
    def __init__(self, cell, ext_conductivity, stim_pos):
        self.cell = cell
        self.num_segments = self.cell.totnsegs
        self.ext_conductivity = ext_conductivity                                     # [S/m]
        self.stim_pos = stim_pos

    def stim_amp_array(self, stim_amp_min, stim_amp_max, num_axons):
        if (num_axons == 1):
            stim_amp = - 10**np.array([stim_amp_min])
        else:
            stim_amp = - 10**np.random.uniform(stim_amp_min, stim_amp_max, num_axons)
        return stim_amp

    def stim_dur_array(self, stim_dur_min, stim_dur_max, num_axons):
        if (num_axons == 1):
            stim_dur = np.array([stim_dur_min])
        else:
            stim_dur = np.random.uniform(stim_dur_min, stim_dur_max, num_axons)
        return stim_dur

    def axon_pos_array(self, start_pos, num_axons):
        np.random.seed = 42
        if (num_axons == 1):
            x_pos = np.array([start_pos[0]])
            y_pos = np.array([start_pos[1]])
            z_pos = np.array([start_pos[2]])
        else:
            x_pos = ((-1)**np.random.randint(2, size = num_axons) * 3 
                    * 10**np.random.uniform(0, 3, num_axons))      
            y_pos = ((-1)**np.random.randint(2, size = num_axons) * 3
                    * 10**np.random.uniform(0, 3, num_axons)) 
            z_pos = ((-1)**np.random.randint(2, size = num_axons) * 3
                    * 10**np.random.uniform(0, 3, num_axons))

        axon_pos_array = np.array([x_pos, y_pos, z_pos])
        return axon_pos_array

    def axon_rot_array(self, num_axons):
        np.random.seed = 42
        if (num_axons == 1):
            x_rot = np.zeros(num_axons)
            y_rot = np.zeros(num_axons)
            z_rot = np.zeros(num_axons)
        else:
            x_rot = np.random.uniform(0, 2 * np.pi, num_axons)
            y_rot = np.random.uniform(0, 2 * np.pi, num_axons)
            z_rot = np.random.uniform(0, 2 * np.pi, num_axons)

        axon_rot_array = np.array([x_rot, y_rot, z_rot])
        return axon_rot_array

    def ext_pot(self, x, y, z):
        ext_conductivity = self.ext_conductivity 
        stim_pos = self.stim_pos
        
        ext_pot = 1 / (4 * np.pi * ext_conductivity) / np.sqrt((x - stim_pos[0])**2 +
                                                               (y - stim_pos[1])**2 +
                                                               (z - stim_pos[2])**2)
        return ext_pot

    def run_simulation(self, i, stim_amp, stim_dur, axon_pos, axon_rot):
        """ Method generating extracellular potential """

        cell = self.cell 
        num_segments = self.num_segments
        stop_time = 20                                                         # [ms]
        time_step_size = 2**(-5)                                               # [ms]
        num_time_steps = int(stop_time / time_step_size + 1) 
        time_vec = np.arange(num_time_steps) * time_step_size
        stim_delay = 5

        cell.set_pos(axon_pos[0][i], axon_pos[1][i], axon_pos[2][i])
        cell.set_rotation(axon_rot[0][i], axon_rot[1][i], axon_rot[2][i])
        
        t0_idx = np.argmin(np.abs(time_vec - stim_delay))
        t1_idx = np.argmin(np.abs(time_vec - stim_delay - stim_dur[i]))
        stim_current = np.zeros(num_time_steps)
        stim_current[t0_idx:t1_idx] = stim_amp[i]

        pot_on_cell = self.ext_pot(cell.x.mean(axis = 1), cell.y.mean(axis = 1), cell.z.mean(axis = 1))
        v_cell_ext = np.zeros((num_segments, num_time_steps))
        v_cell_ext[:, :] = np.dot(pot_on_cell[None, :].T, stim_current[:, None].T) # completes extracellular potential eq. with current

        # Insert external potential as boundary condition in NEURON:
        cell.insert_v_ext(v_cell_ext, time_vec)
        
        # Run simulation, record segment membrane voltages and membrane currents:
        cell.simulate(rec_vmem = True, rec_imem = True)