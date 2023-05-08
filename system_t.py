# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 15:08:50 2023

@author: cnmlt
"""

import numpy as np
import xarray as xr
import imod
import random
from matplotlib import pyplot as plt
import shutil
import random_field as rf
import os 

class DatasetGenerator:
    def __init__(self):
        self.fixed_seed = False
        self.deeponet_dataset = False
        self.modeldirs = []

        # Define fixed variables
        self.nrow = 128
        self.ncol = 128
        self.days_tot = 10

        self.nlay = 1
        self.shape = (self.nlay, self.nrow, self.ncol)

        self.dx = 500.0
        self.dy = 500.0
        self.zmin = -100.0
        self.xmin = 0.0
        self.xmax = self.dx * self.ncol
        self.ymin = 0.0
        self.ymax = abs(self.dy) * self.nrow
        self.dims = ("layer", "y", "x")

        self.layer = np.array([1])  # this is different in case of more layers!
        self.y = np.arange(self.ymin, self.ymax, self.dy)
        self.x = np.arange(self.xmin, self.xmax, self.dx)
        self.coords = {"layer": self.layer, "y": self.y, "x": self.x}

        self.idomain = xr.DataArray(np.ones(self.shape, dtype=int), coords=self.coords, dims=self.dims)
        self.bottom = xr.DataArray([self.zmin], {"layer": self.layer}, ("layer",))

        # Constant head
        constant_head = xr.full_like(self.idomain, np.nan, dtype=float).sel(layer=[1])
        constant_head[..., 0] = 0.0

        constant_head[..., -1] = 0.0
        constant_head[:, 0, :] = 1.0
        constant_head[:, -1, :] = -1.0
        self.constant_head = constant_head

        # Node properties
        self.icelltype = xr.DataArray([1], {"layer": self.layer}, ("layer",))

        # random field
        self.k_min = 0.005
        self.k_max = 0.5



        # Drainage
        self.elevation = xr.full_like(self.idomain.sel(layer=1), np.nan, dtype=float)
        self.conductance = xr.full_like(self.idomain.sel(layer=1), np.nan, dtype=float)
        
        for i in range(2, 10):
            self.elevation[10*i, 1:100] = 99*[1] #np.concatenate(([0, 0], np.arange(2, 100-1, 1)/100))
            self.conductance[10*i, 1:100] = 1.0

        # self.elevation[50, 1:100] = np.concatenate(([0, 0], np.arange(2, 100-1, 1)))
        # self.conductance[50, 1:100] = 1.0
        

        starttime = np.datetime64("2020-01-01 00:00:00")
        # Add first steady-state
        timedelta = np.timedelta64(1, "s")  # 1 second duration for initial steady-state
        starttime_steady = starttime - timedelta
        self.times = [starttime_steady, starttime]

        for i in range(self.days_tot - 1):
            self.times.append(self.times[-1] + np.timedelta64(100, "D"))

        self.n_times = len(self.times) - 1

        # Recharge
        rch_rate_allT = []
        for t in range(self.n_times):
               self.rch_rate = random.uniform(3.0e-8, 0) #3.0e-8 # Recharge rate ($m/s$)
               rch_rate_t = xr.full_like(self.idomain.sel(layer=1), self.rch_rate, dtype=float) 
               rch_rate_allT.append(rch_rate_t)
        
        # rechanrge changing in time
        self.rch_rate = xr.concat(rch_rate_allT, dim="time").assign_coords(time = self.times[:-1])



        self.transient = xr.DataArray(
            [False] + [True] * self.n_times, {"time": self.times}, ("time",)
        )

        
    def well_rate_function(self):
        """
        This function generates a random well pumping rate for each day (segment).
        The well rate is constant within each segment and is chosen randomly from a range.

        Returns:
            well_rate (list): A list containing the well pumping rate for each day
        """

        # Set the minimum and maximum well rate values
        min_rate = -500
        max_rate = -50

        # Define the segment length (in days)
        segment_length = 1

        # Create an empty list to hold the well rate values
        well_rate = []

        # Loop through the number of segments (days)
        for _ in range(self.n_times // segment_length):
            # Choose a random integer for the segment and repeat it
            pumping_rate = float(random.choice(range(min_rate, max_rate)))
            well_rate += [pumping_rate] * segment_length

        # Check if the length of the list is less than n_times
        if len(well_rate) < self.n_times:
            # If it is, calculate the number of times to add the current rate
            times_to_add = self.n_times - len(well_rate)
            # Add the current rate to the list the required number of times
            well_rate += [pumping_rate] * times_to_add

        return well_rate

    def generate_dataset(self, n_samples):
        """
        This function generates a dataset for training and testing, containing inputs and targets.
        The first input is a 2-channel image with the location of a well and permeability K divided into 5 classes.
        The second input is the pumping rate of the well during 10 days.
        The output is the hydraulic head in the 10 days (10 frames).
    
        Args:
            n_samples (int): The number of samples to generate.
    
        Returns:
            dataset_input (np.array): The input dataset containing well locations and permeability K.
            time_series_input (np.array): The input dataset containing well pumping rates.
            dataset_target (np.array): The target dataset containing hydraulic head values.
        """
    
        self.n_samples = n_samples
        split_sample = int(self.n_samples * 0.8)
    
        # Initialize data containers
        self.dataset_input = []
        self.dataset_target = []
        self.time_series_input = []
    
        for i in range(self.n_samples):
            if i % 100 == 0:
                print("\n Generating sample: " + str(i) + "\n=============================================================================")
    
            # Create permeability K array
            K = np.full(shape=(self.nlay, self.nrow, self.ncol), fill_value=0.5)
            K = rf.create_Krf(alpha=-4., discrete=True, minK=self.k_min, maxK=self.k_max, size=self.nrow, fixed_seed=self.fixed_seed)
            K = K.reshape(self.nlay, self.nrow, self.ncol)
            self.K = K
            self.k = xr.DataArray(self.K, coords=self.coords, dims=self.dims)
    
            # Set k33 constant value
            self.k33 = xr.DataArray([2.0e-3], {"layer": self.layer}, ("layer",))
            if self.fixed_seed:
                random.seed(11)
    
            # Set the number of wells
            self.n_wells = 3
    
            # Randomly select well locations (rows and columns)
            self.well_row = random.sample(range(1, self.nrow - 1), self.n_wells)
            self.well_column = random.sample(range(1, self.ncol - 1), self.n_wells)
            self.well_layer = self.n_wells * [1]
            
            # Initialize arrays for well positions and rates
            arr = np.zeros((self.nrow, self.ncol, self.days_tot))
            well_pos = np.zeros((self.nrow, self.ncol, 1))
    
            well_rates = []
            for i in range(self.n_wells):
                # Create well rate function
                well_rate = self.well_rate_function()
    
                # Get well coordinates
                well_row = self.well_row[i]
                well_col = self.well_column[i]
    
                # Assign well rate to the well
                arr[well_row, well_col, :] = well_rate
                well_rates.append(well_rate)
    
                # Set well position
                well_pos[well_row, well_col, :] = 1
    
            # Combine well rates
            well_rate_all_days = [list(values) for values in zip(*well_rates)]
    
            # Run iMOD model and store head values
            head = self.run_imod(well_rate_all_days)
            self.dataset_target.append(head)
    
            self.K = self.K.reshape(self.nrow, self.ncol, 1)
            self.dataset_input.append(np.expand_dims(np.concatenate((self.K, np.array(well_pos)), axis=2), axis=-1))
            self.time_series_input.append(np.array(well_rate_all_days))

        self.time_series_input = np.array(self.time_series_input)
        self.dataset_input = np.array(self.dataset_input)
                    
        # check a sample
        # self.dataset_target[0].isel(layer=0, time=5).plot.contourf()
        
        # merge together and convert from xarray to np
        self.dataset_target = xr.concat(self.dataset_target, dim='sample')
        self.dataset_target = self.dataset_target.to_numpy()[:,:,0,:,:]                   
        self.dataset_target = self.dataset_target.swapaxes(1,-1) #put time as last coordinate
        self.dataset_target = self.dataset_target.swapaxes(1,2) #exchange x and y

        if self.deeponet_dataset:
            x_ = np.linspace(0., 1., self.nrow)
            y_ = np.linspace(0., 1., self.ncol)
            tsteps = np.linspace(0., 1., self.days_tot)
            XX, YY, TT = np.meshgrid(x_, y_, tsteps, indexing='ij')
            y_stacked = np.hstack((XX.flatten()[:,None], YY.flatten()[:,None], TT.flatten()[:,None]))
            y_stacked = y_stacked.reshape(self.nrow,self.ncol,len(tsteps),-1)
            self.trunk   = np.repeat(y_stacked[np.newaxis, :, :, :], self.n_samples, axis=0)
        

        return self.dataset_input, self.time_series_input, self.dataset_target

    
    
    def run_imod(self, well_rate_allT, delete_f = True):

                

        well_rate_gwf = xr.DataArray(well_rate_allT, coords={"time": self.times[:-1], "well_nr": list(range(1, self.n_wells+1))}, dims=("time","well_nr"))
        
        
        # Write the model
        
        gwf_model = imod.mf6.GroundwaterFlowModel()
        gwf_model["dis"] = imod.mf6.StructuredDiscretization(
            top=0.0, bottom=self.bottom, idomain=self.idomain
        )
        gwf_model["chd"] = imod.mf6.ConstantHead(
            self.constant_head, print_input=True, print_flows=True, save_flows=True
        )
        
        gwf_model["ic"] = imod.mf6.InitialConditions(head=0.0)
        gwf_model["npf"] = imod.mf6.NodePropertyFlow(
            icelltype=self.icelltype,
            k=self.k,
            k33=self.k33,
            variable_vertical_conductance=True,
            dewatered=True,
            perched=True,
            save_flows=True,
        )
        gwf_model["sto"] = imod.mf6.SpecificStorage(
            specific_storage=1.0e-6,
            specific_yield=0.15,
            transient=self.transient,
            convertible=0,
        )
        gwf_model["oc"] = imod.mf6.OutputControl(save_head="all", save_budget="all")
        
        gwf_model["rch"] = imod.mf6.Recharge(self.rch_rate)

        gwf_model["wel"] = imod.mf6.WellDisStructured(
            layer=self.well_layer,
            row=self.well_row,
            column=self.well_column,
            rate=well_rate_gwf,
            print_input=True,
            print_flows=True,
            save_flows=True,
        )
        
        gwf_model["drn"] = imod.mf6.Drainage(
            elevation=self.elevation,
            conductance=self.conductance,
            print_input=True,
            print_flows=True,
            save_flows=True,
        )
        
        # Attach it to a simulation
        simulation = imod.mf6.Modflow6Simulation("ex01-twri")
        simulation["GWF_1"] = gwf_model
        # Define solver settings
        simulation["solver"] = imod.mf6.Solution(
            modelnames=["GWF_1"],
            print_option="summary",
            csv_output=False,
            no_ptc=True,
            outer_dvclose=1.0e-4,
            outer_maximum=500,
            under_relaxation=None,
            inner_dvclose=1.0e-4,
            inner_rclose=0.001,
            inner_maximum=100,
            linear_acceleration="cg",
            scaling_method=None,
            reordering_method=None,
            relaxation_factor=0.97,
        )
        # Collect time discretization
        simulation.create_time_discretization(additional_times=self.times)

        modeldir = imod.util.temporary_directory()
        simulation.write(modeldir)
        simulation.run()
        
        # Open the results
        head = imod.mf6.open_hds(
            modeldir / "GWF_1/GWF_1.hds",
            modeldir / "GWF_1/dis.dis.grb",
        )

        self.modeldirs.append(modeldir)
        
        # head.isel(layer=0, time=0).plot.contourf()

        return head


# Create dataset generator
dataset_generator = DatasetGenerator()

# Generate train set
U_train, Y_train, s_train = dataset_generator.generate_dataset(n_samples=6)

# # Generate test set
U_test, Y_test, s_test = dataset_generator.generate_dataset(n_samples=1)

plot_example = True

if plot_example == True:
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    for n_example in range(5):
    
        f1 = U_train[n_example,:,:,1,0]
        f2 = U_train[n_example,:,:,0,0] #permeability
        f3 =  s_train[n_example,:,:,-1]
        
        # Create a figure with two subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 5))
        
        # Plot an image with discrete colorbar in the left subplot
        im = ax1.imshow(f1 , cmap='hot')
        ax1.set_title('Input')
        # cbar = fig.colorbar(im, ax=ax1, ticks=np.unique(U_train[3,:,:,0]))
        
        arr = f1
        # find the row and column indices of non-zero elements
        nonzero_indices = np.where(arr != 0)
        # find the unique non-zero values in the array
        unique_nonzero_values = np.unique(arr[nonzero_indices])
        # print the unique non-zero values and their row and column indices
        print("Unique non-zero values:", unique_nonzero_values)
        
        for value in unique_nonzero_values:
            rows, cols = np.where((arr == value) & (arr != 0))
            print(f"Value {value} found at rows {rows} and cols {cols}")
            for n, p in zip(rows, cols):
            #     print(n,p)
                ax1.text(p, n, "%.0f"%(f1[n,p]), size=10,
                         va="bottom", ha="center", multialignment="left")
    
        
        # Plot an image in the right subplot
        im = ax2.imshow(f2, cmap='afmhot')
        ax2.set_title('Permeability')
        cbar = fig.colorbar(im, ax=ax2)
    
    
        # Plot an image in the right subplot
        im = ax3.imshow(f3, cmap='afmhot')
        cset = ax3.contour(f3, cmap='Set1_r', linewidths=2) # cmap='gray'
        ax3.clabel(cset, inline=False, fmt='%1.2f', fontsize=10, colors = 'k')
        ax3.set_title('Output')
        cbar = fig.colorbar(im, ax=ax3)
        # Display the plot
        plt.show()
    

        # nonzero_indices = np.argwhere(U_train[n_example,:,:, -1] != 0)  # Find indices of non-zero elements in last axis
        # x, y, _ = nonzero_indices.T  # Separate indices into x and y coordinates
        # values = U_train[n_example, nonzero_indices[:, 0], nonzero_indices[:, 1], :]  # Extract non-zero values
        # for i in range(len(values)):
        #     plt.plot(values[i])
        # plt.show()
        

        fig, axs = plt.subplots(nrows=2, ncols=5)
        vmin, vmax = s_train[n_example,:,:, :].min(), s_train[n_example,:,:, :].max()
        # Iterate over subplots and display a frame of the array in each subplot
        for i, ax in enumerate(axs.flat):
            frame = s_train[n_example,:,:, i]  # Extract frame from array
            im = ax.imshow(frame, cmap='afmhot', vmin=vmin, vmax=vmax)
            ax.set_axis_off()
            ax.set_title(f"Frame {i}", size=10)
        
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(im, cax=cbar_ax)
        plt.subplots_adjust(hspace=-0.1, wspace=0.)

        plt.show()

    


for modeldir in dataset_generator.modeldirs:
      shutil.rmtree(modeldir)    
        

common_string = '.npy'

np.save(os.path.join("10video_1well_5000", 'X_train' + common_string), U_train)
np.save(os.path.join("10video_1well_5000", 'Y_train' + common_string),s_train)
np.save(os.path.join("10video_1well_5000", 'X_test' + common_string), U_test)
np.save(os.path.join("10video_1well_5000", 'Y_test' + common_string), s_test)

np.save(os.path.join("10video_1well_5000", 'timeseq_train' + common_string), Y_train)
np.save(os.path.join("10video_1well_5000", 'timeseq_test' + common_string), Y_test)

