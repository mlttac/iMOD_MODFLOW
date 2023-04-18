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

class DatasetGenerator:
    def __init__(self, n_samples):
        self.n_samples = n_samples
        self.modeldirs = []
        # Define fixed variables

        self.nrow = 32 
        self.ncol = 32
        self.days_tot = 100
        
        
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
        constant_head[..., -1] = 0.0
        constant_head[:, 0,:] = 0.0 
        constant_head[:, -1,:] = 0.0 
        self.constant_head = constant_head
        
        self.n_wells = 1
        self.well_layer = [1]

        
        # Node properties
        self.icelltype = xr.DataArray([1], {"layer": self.layer}, ("layer",))
        K  = np.full(shape = (self.nlay, self.nrow,self.ncol), fill_value = 0.05 )
        # Define the indices of the vertices of the triangle
        center_m, center_n = self.nrow // 2, self.ncol // 2
        
        k_values = [ 0.5 , 
                    0.5, 
                    0.5, 
                    0.5]
        
        # divide the rectangular array into four triangles
        K[:,:center_m,:center_n] = k_values[0]
        K[:,:center_m,center_n:] = k_values[1]
        K[:,center_m:,:center_n] = k_values[2]
        K[:,center_m:,center_n:] = k_values[3]
        
        self.K = K
        
        
        # # Print the final array
        # im = plt.imshow(K[0,:,:])
        # plt.colorbar(im)
        self.k = xr.DataArray(self.K, coords=self.coords, dims=self.dims) # Horizontal hydraulic conductivity ($m/day$)
        self.k33 = xr.DataArray([2.0e-3], {"layer": self.layer}, ("layer",))
        
        starttime = np.datetime64("2020-01-01 00:00:00")
        # Add first steady-state
        timedelta = np.timedelta64(1, "s")  # 1 second duration for initial steady-state
        starttime_steady = starttime - timedelta
        self.times = [ starttime_steady , starttime]
        

        for i in range(self.days_tot-1):
            self.times.append(self.times[-1] + np.timedelta64(1, "D"))
        
        self.n_times = len(self.times) - 1
        
        self.transient = xr.DataArray(
            [False] + [True] * self.n_times, {"time": self.times}, ("time",)
        )
        
    def generate_set(self, is_train):
        split_sample = int(self.n_samples * 0.8)

        if is_train:
            # generate data
            self.dataset_branch = []
            self.dataset_target = []
            self.well_locs = []
            
            for i in range(self.n_samples):
                if i%100 == 0:
                    print("\n\Generating sample: " + str(i) + "\n=============================================================================") 
    
    
    
                # self.well_row = [16 ]
                # self.well_column = [16]
                self.well_row = random.sample(range(2, self.nrow-2), self.n_wells)
                self.well_column = random.sample(range(2, self.ncol-2), self.n_wells)

                loc_well = np.zeros((self.nrow, self.ncol))
                loc_well[self.well_row, self.well_column] = 1
                
                # Generate random variable
                # # well rate changing with time
                min_rate = -30
                max_rate = 0
                
                segment_length = 15
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
                    # Add 0 to the list the required number of times
                    well_rate += [p_rate] * times_to_add
                    
                # Generate sample with fixed and random variables
                head = self.run_imod(well_rate)
                
                self.dataset_branch.append(np.array(well_rate))
                # Append sample to dataset
                self.dataset_target.append(head)
                
                self.well_locs.append(loc_well)
                    
            self.dataset_branch = np.array(self.dataset_branch)
            self.well_locs = np.array(self.well_locs)

            # check a sample
            # self.dataset_target[0].isel(layer=0, time=5).plot.contourf()
            
            # merge together and convert from xarray to np
            self.dataset_target = xr.concat(self.dataset_target, dim='sample')
            self.dataset_target = self.dataset_target.to_numpy()[:,:,0,:,:]
                   
            # self.dataset_target = self.dataset_target.reshape(self.n_samples, self.nrow, self.ncol, self.days_tot)
            self.dataset_target = self.dataset_target.swapaxes(1,-1)
            x_ = np.linspace(0., 1., self.nrow)
            y_ = np.linspace(0., 1., self.ncol)
            tsteps = np.linspace(0., 1., self.days_tot)
            XX, YY, TT = np.meshgrid(x_, y_, tsteps, indexing='ij')
            y_stacked = np.hstack((XX.flatten()[:,None], YY.flatten()[:,None], TT.flatten()[:,None]))
            y_stacked = y_stacked.reshape(self.nrow,self.ncol,len(tsteps),3)
            self.trunk   = np.repeat(y_stacked[np.newaxis, :, :, :], self.n_samples, axis=0)
            
            # In this case, the inputs are multi-modal: Pumping profile (time sequence) + Well Location (Image)
            # Output are the frames of Head in time
            return self.dataset_branch[:split_sample], self.well_locs[:split_sample], self.dataset_target[:split_sample]
        else:
            return self.dataset_branch[split_sample:], self.well_locs[split_sample:], self.dataset_target[split_sample:]
    
    
    def run_imod(self, well_rate, delete_f = True):

                
        well_rate_allT = [[x] for x in well_rate]
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
        
        return head


# Create dataset generator
dataset_generator = DatasetGenerator(n_samples=1000)

# Generate train set
U_train, Y_train, s_train = dataset_generator.generate_set(is_train=True)

# Generate test set
U_test, Y_test, s_test = dataset_generator.generate_set(is_train=False)

for modeldir in dataset_generator.modeldirs:
     shutil.rmtree(modeldir)    
        

outputName = "well_t_loc-1000.npz"

np.savez_compressed(outputName, U_train=U_train, Y_train=Y_train, s_train=s_train, 
                    U_test=U_test, Y_test=Y_test, s_test=s_test)

