"""
# .. _MODFLOW6 example problems: https://github.com/MODFLOW-USGS/modflow6-examples
# .. _description: https://modflow6-examples.readthedocs.io/en/master/_examples/ex-gwf-twri.html
# .. _notebook: https://github.com/MODFLOW-USGS/modflow6-examples/tree/master/notebooks/ex-gwf-twri.ipynb
# .. _iMOD Python: https://deltares.gitlab.io/imod/imod-python/index.html
"""

# We'll start with the usual imports. As this is an simple (synthetic)
# structured model, we can make due with few packages.

import numpy as np
import xarray as xr
import pandas as pd
import imod
import random

import os
path =  r'C:/codes/DeepONet/data_generation'
os.chdir(path)

import sys
myModules = 'FDsolver'
if not myModules in sys.path:
    sys.path.insert(0, myModules)

    
import mfgrid
import math
import random_field as rf

from matplotlib import pyplot as plt
# import matplotlib.ticker as mtick
from matplotlib.ticker import FormatStrFormatter
from matplotlib import cm

# %%
# Plotting
       



def plotfigs(f1,title1,f2,title2,contour1= False, contour2= True, sharedColorbar12 = False):

       
       fig, _axs = plt.subplots(nrows=1, ncols=2, figsize=(25,6))
       axs = _axs.flatten()
        
       ax1 = axs[0]
       ax2 = axs[1]


       # Plot 1
       cmap = cm.viridis
       # classes = np.unique(k)


       cntr1 = ax1.imshow(f1 , 
                          cmap=cmap, 
                          # norm = norm,
                           # cmap=cmap,
                           extent=[0,1,0,1],
                           # interpolation='quadric', 
                           origin='lower', 
                           aspect='auto')
       if contour1== True:
              cset = ax1.contour(f1, cmap='Set1_r', linewidths=2,extent=[0,1,0,1]) # cmap='gray'
              ax1.clabel(cset, inline=False, fmt='%1.2f', fontsize=10, colors = 'k')
       fig.colorbar(cntr1, ax=ax1, cmap=cmap)# ticks=classes) # , format='%1i'
       # ax1.locator_params(axis='y', nbins=3)
       # ax1.set(xlim=(0, 1), ylim=(0, 1))
       ax1.set_xlabel("$x_1$")
       ax1.set_ylabel("$x_2$")
       

       # ax1.ticklabel_format(axis="y", style="plain", scilimits=(0,0))
       ax1.set_title(title1)
       
       # Plot 2
       vmin_2 , vmax_2, levels_2 = ( f1.min(),  f1.max(), cset.levels ) if sharedColorbar12 == True else ( f2.min(),  f2.max() , None)

       cntr2 = ax2.imshow(f2 ,  
                         cmap= 'afmhot', 
                          vmin=vmin_2, vmax=vmax_2,
             extent=[0,1,0,1],
              interpolation='quadric', 
             origin='lower', 
             aspect='auto')
       if contour2== True:
              cset = ax2.contour(f2, cmap='Set1_r', linewidths=2,extent=[0,1,0,1], levels=levels_2) # cmap='gray'
              ax2.clabel(cset, inline=True, fmt='%1.2f', fontsize=10, colors = 'k')
       fig.colorbar(cntr2, ax=ax2)
       ax2.set(xlim=(0, 1), ylim=(0, 1))
       ax2.set_xlabel("$x_1$")
       ax2.set_ylabel("$x_2$")
       ax2.ticklabel_format(axis="y", style="plain", scilimits=(0,0))
       ax2.set_title(title2)
       

       return
              



# %%
# Create grid coordinates
def run(z_drain_top = 0, cond_drain = 0.010, n_drains = 11, spacing_drains=1, k_value = None, wells_active = False):
       nlay = 1 # Number of layers
       nrow = 32 # Number of columns
       ncol = 32 # Number of columns
       dx = 1000 # Column width ($m$)
       dy = -1000  # Row width ($m$)
       zmin = -60.0 # Layer bottom elevations ($m$)
       zmax = 0.0 # Top of the model ($m$)
       
       # Boundary and Initial conditions
       BC_left = 0
       BC_right = 0
       BC_up = 0
       BC_down = 0 
       HI = 0.0 # Initial Head 
       
       # Properties of the aquifer
       S = 0.05 # Elastic storage[-]
       Ss = S / (zmax-zmin) #3.2e-4
       


       shape = (nlay, nrow, ncol)
       
       xmin = 0.0
       xmax = dx * ncol
       ymin = 0.0
       ymax = abs(dy) * nrow
       dims = ("layer", "y", "x")
       
       layer = np.array([1])
       y = np.arange(ymax, ymin, dy) + 0.5 * dy
       x = np.arange(xmin, xmax, dx) + 0.5 * dx
       coords = {"layer": layer, "y": y, "x": x}
       
       # Create DataArrays
       
       idomain = xr.DataArray(np.ones(shape, dtype=int), coords=coords, dims=dims)
       bottom = xr.DataArray([zmin], {"layer": layer}, ("layer",))
       
       # Constant head (Boundary conditions)
       constant_head = xr.full_like(idomain, np.nan, dtype=float).sel(layer=[1])
       constant_head[..., 0] = BC_left 
       constant_head[..., -1] = BC_right 
       constant_head[:, 0,:] = BC_up 
       constant_head[:, -1,:] = BC_down 
       # constant_head.sel(layer=1).plot.imshow()
       
       # Drainage
       elevation = xr.full_like(idomain.sel(layer=1), np.nan, dtype=float)
       conductance = xr.full_like(idomain.sel(layer=1), np.nan, dtype=float)
       
       # n_drains = 5
       y_drain = 16 # start from the top
       x_drain = 12 # start from the left
       z_drain_top = -4
       dz_lowerDrains = np.arange(0, n_drains)
       # cond_drain = 1
       
       elevation[:,x_drain] = z_drain_top
       # elevation.plot.imshow()
       conductance[:,x_drain] = cond_drain
       
       # add a second drain
       y_drain2 = 25 
       x_drain2 = 20 
       elevation2 = xr.full_like(idomain.sel(layer=1), np.nan, dtype=float)
       conductance2 = xr.full_like(idomain.sel(layer=1), np.nan, dtype=float)
       elevation2[:,x_drain2] = z_drain_top
       conductance2[:,x_drain2] = cond_drain
       
       drain_active = True
       
       # Time:  two stress periods, the first ending at 0.5d and the second ending at 200d :
       start_date = pd.to_datetime("2020-01-01")
       duration = pd.to_timedelta(["0", "200d"]) #duration = pd.to_timedelta(["0", "0.5d", "200d"])
       
       # Collect time discretization
       times = start_date + duration
       
       
       
       if wells_active == True:
              Q_well = -0.5  #m2/day
              well_layer = [1]
              well_row = [16]
              well_column = [16]
              n_wells = len(well_column)

              #fix well rate in time
              well_rate = [Q_well] * n_wells


       # Node properties
       icelltype = xr.DataArray([1], {"layer": layer}, ("layer",))

       # k_value_domain = random.choice(np.geomspace(0.0001, 0.005, num=100))
       # k_value_inside = random.choice(np.geomspace(0.0001, 0.005, num=100))
       # k_value_inside2 = random.choice(np.geomspace(0.0001, 0.005, num=100))
       # K  = np.full(shape = (nlay, nrow,ncol), fill_value = k_value_domain )
       # y_start = random.choice(np.arange(1, y_drain-3,1))
       # y_end = random.choice(np.arange(y_drain+3,nrow-1, 1))
       # x_start = random.choice(np.arange(1, x_drain-3,1))
       # x_end = random.choice(np.arange(x_drain+3,nrow-1, 1))
       # K[:, y_start : y_end  ,x_start: x_end] = k_value_inside 
       # K[:, y_drain-4 : y_drain+4  ,x_drain-4 : x_drain+4] = k_value_inside
       # y_start = random.choice(np.arange(1, y_drain2-3,1))
       # y_end = random.choice(np.arange(y_drain2+3,nrow-1, 1))
       # x_start = random.choice(np.arange(1, x_drain2-3,1))
       # x_end = random.choice(np.arange(x_drain2+3,nrow-1, 1))
       # K[:, y_drain2-4 : y_drain2+4  ,x_drain2-4 : x_drain2+4] = k_value_inside2
              

       # # random field

       k1 = 0.0001
       k2 = 0.005
       K = rf.create_Krf(alpha=-3.0, discrete = False, minK = k1 , maxK = k2, n_classes = 5, size = nrow ,  
                         reshuffle_k = False,
                         log_shape = True) 
       
       
       
       # # K = rf.split_chunks(size = 32 , chunk_size = 4, mink= 0.0001, maxk=0.005, log_intervals = True)

       K = K.reshape(nlay, nrow,ncol)

       # test to see effect of the boundary (only one drain)
       # K  = np.full(shape = (nlay, nrow,ncol), fill_value = 0.005 )
       # K[:,y_drain-6 : y_drain-4  ,x_drain-6 : x_drain+6] = 0.0001
       # K[:,y_drain+4 : y_drain+6  ,x_drain-6 : x_drain+6] = 0.0001   
       # K[:,y_drain-6 : y_drain+6  ,x_drain-6 : x_drain-4] = 0.0001
       # K[:,y_drain-6 : y_drain+6  ,x_drain+4 : x_drain+6] = 0.0001
       
       # K  = np.full(shape = (nlay, nrow,ncol), fill_value = k_value_domain )
       
       k = xr.DataArray(K, coords=coords, dims=dims) # Horizontal hydraulic conductivity ($m/day$)
       k33 = k # Vertical hydraulic conductivity ($m/day$)
       
       # Write the model
       # Define an empty model, the add  parameters and boundary conditions 
       
       gwf_model = imod.mf6.GroundwaterFlowModel()
       gwf_model["dis"] = imod.mf6.StructuredDiscretization(
           top=zmax, bottom=bottom, idomain=idomain
       )
       gwf_model["chd"] = imod.mf6.ConstantHead(
           constant_head, print_input=True, print_flows=True, save_flows=True
       )
       if drain_active:
              for count, dz in enumerate(dz_lowerDrains):
                     gwf_model["drn_"+str(count)] = imod.mf6.Drainage(
                         elevation=elevation,# - dz*spacing_drains,
                         conductance=conductance,
                         print_input=True,
                         print_flows=True,
                         save_flows=True,
                     )
                            
              # # second drain
              # for count, dz in enumerate(dz_lowerDrains):
              # gwf_model["drn2_"+str(count)] = imod.mf6.Drainage(
              #     elevation=elevation2,
              #     conductance=conductance2,
              #     print_input=True,
              #     print_flows=True,
              #     save_flows=True,
              #         )
              
              # gwf_model["drn_2"] = imod.mf6.Drainage(
              #     elevation=elevation-3,
              #     conductance=conductance,
              #     print_input=True,
              #     print_flows=True,
              #     save_flows=True,
              # )
       
       gwf_model["ic"] = imod.mf6.InitialConditions(head=HI) 
       
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
           specific_storage=Ss,
           specific_yield=0.15,
           transient=False,
           convertible=0,
       )
       gwf_model["oc"] = imod.mf6.OutputControl(save_head="all", save_budget="all")
       # gwf_model["rch"] = imod.mf6.Recharge(rch_rate)
       
       if wells_active: 
              gwf_model["wel"] = imod.mf6.WellDisStructured(
                  layer=well_layer,
                  row=well_row,
                  column=well_column,
                  rate=well_rate,
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
       
       
       
       # times = [start_date, start_date + duration]
       simulation.create_time_discretization(additional_times=times)
       
       # simulation.create_time_discretization(
       #     additional_times=["2000-01-01", "2000-01-02", "2000-01-03", "2000-01-04"]
       # )

       # We'll create a new directory in which we will write and run the model.
       
       
       modeldir = imod.util.temporary_directory()
       
       # if I want to store the results somewhere
       # from pathlib import Path
       # modeldir = Path('C:/Users/cnmlt/AppData/Local/imod-python')
       
       simulation.write(modeldir)

       # Run the model
       
       simulation.run( )

       # Open the results
       # ----------------
       #
       # We'll open the heads (.hds) file.
       
       head = imod.mf6.open_hds(
           modeldir / "GWF_1/GWF_1.hds",
           modeldir / "GWF_1/dis.dis.grb",
       )
       
       cbc = imod.mf6.open_cbc(
           modeldir / "GWF_1/GWF_1.cbc",
           modeldir / "GWF_1/dis.dis.grb",
       )
       

       # Visualize the results
       # ---------------------
       
       # head.isel(layer=0, time=0).plot.contourf()
       
       # data at the drain
       # xr.merge([cbc]).compute().isel(y=y_drain, time=-1, layer=0)["drn"]
       # xr.merge([cbc]).compute().isel(y=y_drain, time=-1, layer=0)[["drn", "drn_2"]]
       
       # cbc["chd"].isel(y=y_drain, time=-1, layer=0).plot()
       
       # # %%
       # # Store results
       # head.isel(time=1).values
       
       
       # head.isel(y=y_drain, layer=0, time=-1).plot()
       
       # head.isel(layer=0, x=x_drain, y=y_drain).plot()
       # or get the values: 
       return K , head

# to store the results I can actually use the xarrays functionalities:
# example: xr.concat([head, head], dim="scenario")

k_value = 0.005

k , head =  run(k_value = k_value, wells_active = True)

S = head.isel(layer=0, time=0).values

  

plotfigs(f1 = k.reshape(32,32),
         title1 = 'Hydraulic conductivity',
         f2 = S.reshape(32,32),
         title2 = 'Hydraulic Head')

       # %%

pre_analysis = False

# if pre_analysis is True:
k = 0.5
z_drain_top = 0
cond_drain = 0.01000
n_drains = 11
spacing_drains = 1
h_res= []
# x_analysis = np.arange( 1, 11)/10
x_analysis = np.geomspace(0.0001, 0.005, num=10)
# x_analysis = [0.1]
# x_analysis = np.arange(-10, 0) #np.linspace(0, 10, num=10)
y_drain = 15 # start from the top
x_drain = 20 # start from the left
for i in x_analysis:
       K, hres_n = run( k_value = i, wells_active = True) #z_drain_top, cond_drain, n_drains, spacing_drains, 
       h_res.append(hres_n.isel(layer=0, x=x_drain, y=y_drain).values)


       
dz_lowerDrains = np.arange(0, n_drains)
elevation_dr = []
for count, dz in enumerate(dz_lowerDrains):
           elevation_dr.append(z_drain_top - dz*spacing_drains)
       
fig = plt.figure()

ax = fig.add_subplot(111)
param = 'Hydraulic conductivity'
fig.suptitle(param + '\n n.drains={},   ztop={}m,  C drains={}'.format( n_drains, z_drain_top, cond_drain), fontsize=12)

ax.scatter(x_analysis, h_res)
for i in elevation_dr:
       ax.axhline( y = i, color ='g', linestyle = '--')
# ax.ticklabel_format(style='sci', axis='x', scilimits=(-3,4))
ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
ax.xaxis.set_ticks(x_analysis)
ax.tick_params(axis='x', rotation=90)
ax.set_ylabel('Head at the drain [m]', fontsize=12)
ax.set_xlabel(param + '[m/d]', fontsize=12)
plt.show()




#%% Dataset creation

if __name__ == "__main__":
       
       # DEFINE THE INPUTS:
       outputName = "../data/1well_center_1Hordrain_rfK.npz"
       num_train = 1000
       num_test = 100
       s = 32
       dy = 2 #(x,y)
       du = 1
       Nt= 1
  
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
             






              for iter in range(realisation[0]):
                     # print("\n\nRunning Iter: " + str(realisation) + "\n=============================================================================")

                     count +=1
                     k_value = random.choice(np.geomspace(0.0001, 0.005, num=10))
                     k , head =  run(k_value = k_value, wells_active = True)
                     
                     S = head.isel(layer=0, time=0).values
                     
                     folder[0].append(k) 
                     # folder[1].append(Y_2d) #(32, 32, 10, 3)
                     # S = np.einsum('kij->ijk', S)
                     folder[2].append(S.reshape(s,s))

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
