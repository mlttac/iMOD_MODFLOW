# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 14:14:30 2023

@author: cnmlt
"""

"""

"""
# %%

import numpy as np
import xarray as xr
import imod
import random
from matplotlib import pyplot as plt
import shutil

# %%

def run(nrow = 32 , ncol = 32, days_tot = 100 , delete_f = True):
    nlay = 1
    
    shape = (nlay, nrow, ncol)
    
    dx = 500.0
    dy = 500.0
    zmin = -100.0
    xmin = 0.0
    xmax = dx * ncol
    ymin = 0.0
    ymax = abs(dy) * nrow
    dims = ("layer", "y", "x")
    
    layer = np.array([1]) # this is different in case of more layers!
    y = np.arange(ymin,ymax, dy)
    x = np.arange(xmin, xmax, dx)
    coords = {"layer": layer, "y": y, "x": x}
    
    idomain = xr.DataArray(np.ones(shape, dtype=int), coords=coords, dims=dims)
    bottom = xr.DataArray([zmin], {"layer": layer}, ("layer",))
    
    # Constant head
    constant_head = xr.full_like(idomain, np.nan, dtype=float).sel(layer=[1])
    constant_head[..., 0] = 0.0
    constant_head[..., -1] = 0.0
    constant_head[:, 0,:] = 0.0 
    constant_head[:, -1,:] = 0.0 
    
    n_wells = 1
    well_layer = [1]
    well_row = [16 ]
    well_column = [16]
    
    # Node properties
    icelltype = xr.DataArray([1], {"layer": layer}, ("layer",))
    K  = np.full(shape = (nlay, nrow,ncol), fill_value = 0.05 )
    # Define the indices of the vertices of the triangle
    center_m, center_n = nrow // 2, ncol // 2
    
    k_values = [ 0.5 , 
                0.5, 
                0.5, 
                0.5]
    
    # divide the rectangular array into four triangles
    K[:,:center_m,:center_n] = k_values[0]
    K[:,:center_m,center_n:] = k_values[1]
    K[:,center_m:,:center_n] = k_values[2]
    K[:,center_m:,center_n:] = k_values[3]
    
    
    # Print the final array
    im = plt.imshow(K[0,:,:])
    plt.colorbar(im)
    k = xr.DataArray(K, coords=coords, dims=dims) # Horizontal hydraulic conductivity ($m/day$)
    k33 = xr.DataArray([2.0e-3], {"layer": layer}, ("layer",))
    
    starttime = np.datetime64("2020-01-01 00:00:00")
    # Add first steady-state
    timedelta = np.timedelta64(1, "s")  # 1 second duration for initial steady-state
    starttime_steady = starttime - timedelta
    times = [ starttime_steady , starttime]
    

    for i in range(days_tot):
        times.append(times[-1] + np.timedelta64(1, "D"))
    
    n_times = len(times) - 1
    
    transient = xr.DataArray(
        [False] + [True] * n_times, {"time": times}, ("time",)
    )
    
    # # well rate changing with time
    min_rate = -30
    max_rate = 0
    
    segment_length = 15
    # Create an empty list to hold the piecewise function
    well_rate = []
    
    # Loop through the number of segments
    for i in range(n_times//segment_length):
        # Choose a random integer for the segment and repeat it
        p_rate = float(random.choice(range(min_rate, max_rate))) 
        well_rate += [p_rate] * segment_length
    
    # Check if the length of the list is less than n_times
    if len(well_rate) < n_times:
        # If it is, calculate the number of times to add 0
        times_to_add = n_times - len(well_rate)
        # Add 0 to the list the required number of times
        well_rate += [p_rate] * times_to_add
    
    well_rate_allT = [[x] for x in well_rate]
    well_rate_gwf = xr.DataArray(well_rate_allT, coords={"time": times[:-1], "well_nr": list(range(1, n_wells+1))}, dims=("time","well_nr"))
    
    times_plot = range(n_times)

    
    # Write the model
    
    gwf_model = imod.mf6.GroundwaterFlowModel()
    gwf_model["dis"] = imod.mf6.StructuredDiscretization(
        top=0.0, bottom=bottom, idomain=idomain
    )
    gwf_model["chd"] = imod.mf6.ConstantHead(
        constant_head, print_input=True, print_flows=True, save_flows=True
    )
    
    gwf_model["ic"] = imod.mf6.InitialConditions(head=0.0)
    gwf_model["npf"] = imod.mf6.NodePropertyFlow(
        icelltype=icelltype,
        k=k,
        k33=k33,
        variable_vertical_conductance=True,
        dewatered=True,
        perched=True,
        save_flows=True,
    )
    gwf_model["sto"] = imod.mf6.SpecificStorage(
        specific_storage=1.0e-6,
        specific_yield=0.15,
        transient=transient,
        convertible=0,
    )
    gwf_model["oc"] = imod.mf6.OutputControl(save_head="all", save_budget="all")
    gwf_model["wel"] = imod.mf6.WellDisStructured(
        layer=well_layer,
        row=well_row,
        column=well_column,
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
    simulation.create_time_discretization(additional_times=times)

    
    modeldir = imod.util.temporary_directory()
    simulation.write(modeldir)
    simulation.run()
    
    # Open the results
    head = imod.mf6.open_hds(
        modeldir / "GWF_1/GWF_1.hds",
        modeldir / "GWF_1/dis.dis.grb",
    )
    if delete_f == True:
        shutil.rmtree(modeldir)    
    
    return np.array(well_rate), head

# Visualize the results
# ---------------------
well_p, head = run(delete_f = False)

fig = plt.figure()

plt.plot(well_p)
plt.title('Pumping rates in time')
plt.legend()
plt.show()
head.isel(layer=0, x=12, y=16).plot()
hpoint_arr = head.isel(layer=0, x=10, y=16).compute().data

# head.isel(layer=0, time=100).plot.contourf()

for i in np.linspace(0,100,5).round().astype(int):
        fig = plt.figure()
        head.isel(layer=0, time=i).plot.contourf()
    
#%% Dataset creation

if __name__ == "__main__":
       
       # DEFINE THE INPUTS:
       outputName = "../data/well.npz"
       num_train = 1
       num_test = 1
  
       realisations = [[num_train, "Train"], [num_test, "Test"]]  

       U_train = []
       Y_train = []
       s_train = []
       U_test = []
       Y_test = []
       s_test = []
       folders = [[U_train, Y_train, s_train], [U_test, Y_test, s_test]]
       count = 0
       
       for realisation, folder in zip(realisations, folders):
              heads =[]

              for iter in range(realisation[0]):
                     print("\n\nRunning Iter: " + str(realisation) + "\n=============================================================================")

                     count +=1
                     well_p, head = run()
                     folder[0].append(well_p) 
                     heads.append(head)
                   
              #  restructure from xarrray to np
              head_dataset = xr.concat(heads, dim='sample')
              head_dataset_np = head_dataset.to_numpy()[:,:,0,:,:]
              
              folder[2].append(head_dataset_np.reshape(s,s))

       x_ = np.linspace(0., 1., s)
       y_ = np.linspace(0., 1., s)
       XX, YY = np.meshgrid(x_, y_)
       y_stacked = np.hstack((XX.flatten()[:,None], YY.flatten()[:,None]))
       Y_train   = np.repeat(y_stacked[np.newaxis, :, :], num_train, axis=0)
       Y_test   = np.repeat(y_stacked[np.newaxis, :, :], num_test, axis=0)


       if Nt!= 1:
              tsteps = np.linspace(0., 1., 5)
              XX, YY, TT = np.meshgrid(x_, y_, tsteps, indexing='ij')
              y_stacked = np.hstack((XX.flatten()[:,None], YY.flatten()[:,None], TT.flatten()[:,None]))
              y_stacked = y_stacked.reshape(s*s,len(tsteps),3)
              Y_train   = np.repeat(y_stacked[np.newaxis, :, :, :], num_train, axis=0)
              Y_test   = np.repeat(y_stacked[np.newaxis, :, :], num_test, axis=0)
       else: 
              x_ = np.linspace(0., 1., s)
              y_ = np.linspace(0., 1., s)
              XX, YY = np.meshgrid(x_, y_)
              y_stacked = np.hstack((XX.flatten()[:,None], YY.flatten()[:,None]))
              Y_train   = np.repeat(y_stacked[np.newaxis, :, :], num_train, axis=0)
              Y_test   = np.repeat(y_stacked[np.newaxis, :, :], num_test, axis=0)
              
                                   
       np.savez_compressed(outputName, U_train=np.array(U_train).reshape(num_train, s*s, du), 
                           Y_train=np.array(Y_train).reshape(num_train, s*s, dy),
                           s_train=np.array(s_train),
                           U_test=np.array(U_test).reshape(num_test, s*s, du), 
                                               Y_test=np.array(Y_test).reshape(num_test, s*s, dy),
                                               s_test=np.array(s_test))



U_train=np.array(U_train).reshape(num_train, s, s)
s_train=np.array(s_train).reshape(num_train, s,s)

for i in range(5):
       plotfigs(f1 =U_train[i, :, :] ,
                title1 = 'Hydraulic conductivity',
                f2 = s_train[i, :, :],
                title2 = 'Hydraulic Head')

