"""

"""
# %%

import numpy as np
import xarray as xr
import pandas as pd
import imod
import random
from matplotlib import pyplot as plt

# %%

nlay = 1
nrow = 32
ncol = 32
shape = (nlay, nrow, ncol)

# dx = 500.0
# dy = -500.0
# zmin = -100.0
# xmin = 0.0
# xmax = dx * ncol
# ymin = 0.0
# ymax = abs(dy) * nrow
# dims = ("layer", "y", "x")

# layer = np.array([1]) # this is different in case of more layers!
# y = np.arange(ymax, ymin, dy) + 0.5 * dy
# x = np.arange(xmin, xmax, dx) + 0.5 * dx
# coords = {"layer": layer, "y": y, "x": x}




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

# %%

idomain = xr.DataArray(np.ones(shape, dtype=int), coords=coords, dims=dims)

bottom = xr.DataArray([zmin], {"layer": layer}, ("layer",))


# Constant head
constant_head = xr.full_like(idomain, np.nan, dtype=float).sel(layer=[1])
constant_head[..., 0] = 0.0
constant_head[..., -1] = 0.0
constant_head[:, 0,:] = 0.0 
constant_head[:, -1,:] = 0.0 


# Well
# n_wells = 4
# well_layer = [1, 1 , 1 , 1]
# well_row = [5, 15 , 11, 5]
# well_column = [11, 15, 7, 7]

n_wells = 1
well_layer = [1]
well_row = [16 ]
well_column = [16]


# Node properties
icelltype = xr.DataArray([1], {"layer": layer}, ("layer",))
k = xr.DataArray([1.0e-3], {"layer": layer}, ("layer",))
k33 = xr.DataArray([2.0e-3], {"layer": layer}, ("layer",))

# # # random field#
# import random_field as rf
# k1 = 0.0005
# k2 = 0.005

# K = rf.create_Krf(discrete = False, minK = k1 , maxK = k2, n_classes = 5, size = nrow ,  
#                   reshuffle_k = False,
#                   log_shape = True) 


# K = K.reshape(nlay, nrow,ncol)

# test to see effect of the boundary (only one drain)
K  = np.full(shape = (nlay, nrow,ncol), fill_value = 0.05 )

# Define the indices of the vertices of the triangle
row1, col1 = nrow // 2, ncol // 2
row2, col2 = nrow - 1, 0
row3, col3 = nrow - 1, ncol - 1

# Loop through the rows and columns of the array
for i in range(nrow):
    for j in range(ncol):
        # Check if the current indices are within the triangle
        if (row2 - row1) * (j - col1) >= (col2 - col1) * (i - row1) and \
           (row3 - row2) * (j - col2) >= (col3 - col2) * (i - row2) and \
           (row1 - row3) * (j - col3) >= (col1 - col3) * (i - row3):
            # If they are, set the current element to 5
            K[:, i, j] = 0.0001

# Print the final array
im = plt.imshow(K[0,:,:])
plt.colorbar(im)
k = xr.DataArray(K, coords=coords, dims=dims) # Horizontal hydraulic conductivity ($m/day$)
k33 = xr.DataArray([2.0e-3], {"layer": layer}, ("layer",))
 # Vertical hydraulic conductivity ($m/day$)

# start_date = pd.to_datetime("2020-01-01")
# n_times = 3
# final_time = 200
# all_times = np.linspace(10, final_time, n_times+1)
# duration = pd.to_timedelta([str(s) + 'd' if s!=0 else str(int(s)) for s in all_times ])
# times = start_date + duration

starttime = np.datetime64("2020-01-01 00:00:00")
# Add first steady-state
timedelta = np.timedelta64(1, "s")  # 1 second duration for initial steady-state
starttime_steady = starttime - timedelta
times = [ starttime_steady , starttime]

for i in range(365):
    times.append(times[-1] + np.timedelta64(1, "D"))

# times =[
#  np.datetime64('2019-12-31T23:59:59'),
#   np.datetime64('2020-01-02T00:00:00'),
#  np.datetime64('2020-01-02T00:00:10'),
#  np.datetime64('2020-01-03T00:00:00'),
#  np.datetime64('2020-01-04T00:00:00')]

n_times = len(times) - 1

transient = xr.DataArray(
    [False] + [True] * n_times, {"time": times}, ("time",)
)


# # well rate changing with time
min_rate = -30
max_rate = 0

# well_rate_allT = []
# for t in range(n_times):
#         # pumping rate for the current time
#         well_rate_t = [random.choice(range(min_rate, max_rate)) for _ in range(n_wells)]
#         # convert to float
#         well_rate_t = [float(i) for i in well_rate_t]
#         well_rate_allT.append(well_rate_t)

# # Define the number of segments and the length of each segment

segment_length = 30

# Create an empty list to hold the piecewise function
well_rate = []

# well_rate += [0.] * 10
# well_rate += [-30.0] * 90
# well_rate += [0.] * 200
# well_rate += [-30.0] * 50
# p_rate = -15.0

# Loop through the number of segments
for i in range(n_times//segment_length):
    # Choose a random integer for the segment and repeat it
    p_rate = float(random.choice(range(min_rate, max_rate))) if i < 6 or i > 8 else 0
    # p_rate = float(min_rate)
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
fig = plt.figure()
for i in range(len(well_rate_allT[0])):
    plt.plot(times_plot,[pt[i] for pt in well_rate_allT],label = 'id %s'%i)
plt.title('Pumping rates in time')
plt.legend()
plt.show()

# %% Write the model

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


# simulation.create_time_discretization(additional_times=[          
#                 '2020-02-01 00:00:00',
#                 '2020-03-01 00:00:00',
#                '2020-03-14 00:00:00',
#                '2020-07-01 00:00:00'])



# %%
examples = 1
datasets = []
for example in range(examples):
    modeldir = imod.util.temporary_directory()
    simulation.write(modeldir)
    simulation.run()

    # Open the results
    head = imod.mf6.open_hds(
        modeldir / "GWF_1/GWF_1.hds",
        modeldir / "GWF_1/dis.dis.grb",
    )
    
    datasets.append(head)
combined = xr.concat(datasets, dim='sample')
combined_np = combined.to_numpy()[:,:,0,:,:]

np.array(well_rate)

# Visualize the results
# ---------------------
head.isel(layer=0, x=12, y=16).plot()
hpoint_arr = head.isel(layer=0, x=10, y=16).compute().data



for i in np.linspace(0,365,5).round().astype(int):
        fig = plt.figure()
        head.isel(layer=0, time=i).plot.contourf()

# %%
# Assign dates to head
# --------------------
#
# MODFLOW6 has no concept of a calendar, so the output is not labelled only
# in terms of "time since start" in floating point numbers. For this model
# the time unit is days and we can assign a date coordinate as follows:

starttime = pd.to_datetime("2000-01-01")
timedelta = pd.to_timedelta(head["time"], "D")
hds = head.assign_coords(time=starttime + timedelta)

# resample to monthly frequency
hdsm = hds.resample(time="M").mean()
hdsm.plot(col="time", col_wrap=2);
# hdsm.plot.contour(col="time", levels=20, add_colorbar=True);

# %% Extract head at points
x = [9000.0, 9000.0]
y = [9000.0, 9000.0]
selection = imod.select.points_values(hds, x=x, y=y)

#  converted into a variety of tabular file formats.
dataframe = selection.to_dataframe().reset_index()
dataframe = dataframe.rename(columns={"index": "id"})
dataframe

# %% Analysis without xarray
plt.figure()
tstep = 10
plt.pcolormesh( idomain.x.data, idomain.y.data, hds.data[ tstep ,0, :, :]);


# %% Analysis with xarray

head.isel(time=tstep).plot()
head.mean(dim="time").plot()
# hds.sel(time=slice("2020-01-01", "2020-05-01"))
head.isel(time=0, y=2, x=3)  #  much better than ds.air[0, 2, 3]

# get array
head.data[tstep ,0, :, :]
head.compute(layer=0).shape
head.compute()[tstep ,0, :, :]