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
        # self.n_samples = n_samples
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
        
        self.layer = np.array([1]) # this is different in case of more layers!
        self.y = np.arange(self.ymin,self.ymax, self.dy)
        self.x = np.arange(self.xmin, self.xmax, self.dx)
        self.coords = {"layer": self.layer, "y": self.y, "x": self.x}
        
        self.idomain = xr.DataArray(np.ones(self.shape, dtype=int), coords=self.coords, dims=self.dims)
        self.bottom = xr.DataArray([self.zmin], {"layer": self.layer}, ("layer",))
        
        # Constant head
        constant_head = xr.full_like(self.idomain, np.nan, dtype=float).sel(layer=[1])
        constant_head[..., 0] = 0.0
        # constant_head[..., -1] = -2.0
        # constant_head[:, 0,:] = 0.0 
        # constant_head[:, -1,:] = 0.0 
        self.constant_head = constant_head
        
        
        # Node properties
        self.icelltype = xr.DataArray([1], {"layer": self.layer}, ("layer",))
       
        # # # random field#
        self.k_min = 0.005
        self.k_max = 0.5

        # K  = np.full(shape = (self.nlay, self.nrow,self.ncol), fill_value = 0.5 )
        # K = rf.create_Krf(alpha=-4., discrete = True, minK = self.k_min , maxK = self.k_max,size = self.nrow ) 
        # K = K.reshape(self.nlay, self.nrow,self.ncol)
        # self.K = K
        # # Print the final array
        # im = plt.imshow(K[0,:,:])
        # plt.colorbar(im)

        
        starttime = np.datetime64("2020-01-01 00:00:00")
        # Add first steady-state
        timedelta = np.timedelta64(1, "s")  # 1 second duration for initial steady-state
        starttime_steady = starttime - timedelta
        self.times = [ starttime_steady , starttime]
        

        for i in range(self.days_tot-1):
            self.times.append(self.times[-1] + np.timedelta64(100, "D"))
        
        self.n_times = len(self.times) - 1
        
        self.transient = xr.DataArray(
            [False] + [True] * self.n_times, {"time": self.times}, ("time",)
        )
        
        
    def well_rate_fun(self):
                # Generate random variable
                # # well rate changing with time
                min_rate = -30
                max_rate = 0
                
                segment_length = 1
                # Create an empty list to hold the piecewise function
                well_rate = []
                
                # Loop through the number of segments
                for i in range(self.n_times//segment_length):
                    # Choose a random integer for the segment and repeat it
                    p_rate = float(random.choice(range(min_rate, max_rate))) 
                    well_rate += [p_rate] * segment_length
                
                # Check if the length of the list is less than n_times
                if len(well_rate) < self.n_times:
                    # If it is, calculate the number of times to add 0
                    times_to_add = self.n_times - len(well_rate)
                    # p_rate = -15.
                    # Add 0 to the list the required number of times
                    well_rate += [p_rate] * times_to_add
                return well_rate
                  
            
    def generate_set(self, n_samples):
        self.n_samples = n_samples

        split_sample = int(self.n_samples * 0.8)

        # generate data
        self.dataset_input = []
        self.dataset_target = []
        self.time_series_input = []
        
        for i in range(self.n_samples):
            if i%100 == 0:
                print("\n Generating sample: " + str(i) + "\n=============================================================================") 

    
            K  = np.full(shape = (self.nlay, self.nrow,self.ncol), fill_value = 0.5 )
            
            K = rf.create_Krf(alpha=-4., discrete = True, minK = self.k_min , maxK = self.k_max,size = self.nrow, fixed_seed = self.fixed_seed ) 
            K = K.reshape(self.nlay, self.nrow,self.ncol)

            # K  = np.full(shape = (self.nlay, self.nrow,self.ncol), fill_value = 0.05 )
            
            self.K = K
            self.k = xr.DataArray(self.K, coords=self.coords, dims=self.dims) # Horizontal hydraulic conductivity ($m/day$)
            

            self.k33 = xr.DataArray([2.0e-3], {"layer": self.layer}, ("layer",))
            if self.fixed_seed:
                random.seed(11)
            # self.n_wells = random.randint(1, 10) #It includes the last point 
            self.n_wells = 1 #It includes the last point 

            self.well_layer = self.n_wells * [1]

            # using random sample, I have no repetitions: this means that there won't be wells in the same rows or columns 
            self.well_row = random.sample(range(1, self.nrow-1), self.n_wells)
            self.well_column = random.sample(range(1, self.ncol-1), self.n_wells)


            # self.well_row = [74, 74, 55]
            # self.well_column = [62, 74, 2]
            # Initialize the array with zeros
            arr = np.zeros((self.nrow, self.ncol , self.days_tot))
            well_pos = np.zeros((self.nrow, self.ncol, 1))
                                
            well_rates = []
            for i in range(self.n_wells):
                # create rate function
                well_rate = self.well_rate_fun()
                
                #  get coordinates of that well
                wx = self.well_row[i]
                wy = self.well_column[i]
                
                # if i ==0:
                #     well_rate = [-15.] * 10 
                # if i ==1:
                #     well_rate = [-13.] * 10 
                # if i ==2:
                #     well_rate = [-7.] * 10              

                # nv = 5
                # if i ==0:
                #     well_rate[-nv:] = [-15.] * nv 
                # if i ==1:
                #     well_rate[-nv:] = [-13.] * nv
                # if i ==2:
                #     well_rate[-nv:] = [-7.] * nv                              

                #  assign rate to the well
                arr[wx, wy, :] = well_rate
                well_rates.append(well_rate)
                # plt.plot(well_rate)
                
                well_pos[wx, wy,:] = 1
                
            # well_rate_allT = [[x, y, z] for x,y,z in zip(well_rate_1,well_rate_2, well_rate_3)]
            well_rate_allT = [list(values) for values in zip(*well_rates)]
            # print(well_rate_allT[-1])
            
            # Generate sample with fixed and random variables
            head = self.run_imod(well_rate_allT)
            # Append sample to dataset
            self.dataset_target.append(head)

            self.K = self.K.reshape(self.nrow, self.ncol, 1)
            # self.dataset_input.append(np.expand_dims(np.concatenate((self.K, np.array(arr)), axis= 2), axis =-1))
            
            self.dataset_input.append(np.expand_dims(np.concatenate((self.K, np.array(well_pos)), axis= 2), axis =-1))

            self.time_series_input.append(np.array(well_rate_allT))

            
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
        gwf_model["wel"] = imod.mf6.WellDisStructured(
            layer=self.well_layer,
            row=self.well_row,
            column=self.well_column,
            rate=well_rate_gwf,
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
U_train, Y_train, s_train = dataset_generator.generate_set(n_samples=1000)

# dataset_generator = DatasetGenerator(n_samples=200)

# # Generate test set
U_test, Y_test, s_test = dataset_generator.generate_set(n_samples=200)

plot_example = False

if plot_example == True:
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    for n_example in range(2):
    
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
        
        # Iterate over subplots and display a frame of the array in each subplot
        for i, ax in enumerate(axs.flat):
            frame = s_train[n_example,:,:, i]  # Extract frame from array
            ax.imshow(frame)
            ax.set_title(f"Frame {i}")
        
        plt.show()

        



for modeldir in dataset_generator.modeldirs:
      shutil.rmtree(modeldir)    
        



# outputName = "3wells_t-1000.npz"
# # np.savez_compressed(outputName, U_train=U_train, Y_train=Y_train, s_train=s_train, 
#                     U_test=U_test, Y_test=Y_test, s_test=s_test)


common_string = '.npy'

np.save(os.path.join("10video_1well", 'X_train' + common_string), U_train)
np.save(os.path.join("10video_1well", 'Y_train' + common_string),s_train)
np.save(os.path.join("10video_1well", 'X_test' + common_string), U_test)
np.save(os.path.join("10video_1well", 'Y_test' + common_string), s_test)

np.save(os.path.join("10video_1well", 'timeseq_train' + common_string), Y_train)
np.save(os.path.join("10video_1well", 'timeseq_test' + common_string), Y_test)


# np.save('data/X_train.npy', U_train[:,:,:,1,:])
# np.save('Y_train.npy', s_train)
# np.save('X_test.npy', U_test[:,:,:,1,:])
# np.save('Y_test.npy', s_test)

