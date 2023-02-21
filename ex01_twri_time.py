"""
TWRI
====

This example has been converted from the `MODFLOW6 Example problems`_.  See the
`description`_ and the `notebook`_ which uses `FloPy`_ to setup the model.

This example is a modified version of the original MODFLOW example
("`Techniques of Water-Resources Investigation`_" (TWRI)) described in
(`McDonald & Harbaugh, 1988`_) and duplicated in (`Harbaugh & McDonald, 1996`_).
This problem is also is distributed with MODFLOW-2005 (`Harbaugh, 2005`_). The
problem has been modified from a quasi-3D problem, where confining beds are not
explicitly simulated, to an equivalent three-dimensional problem.

In overview, we'll set the following steps:

    * Create a structured grid for a rectangular geometry.
    * Create the xarray DataArrays containg the MODFLOW6 parameters.
    * Feed these arrays into the imod mf6 classes.
    * Write to modflow6 files.
    * Run the model.
    * Open the results back into DataArrays.
    * Visualize the results.

"""
# %%
# We'll start with the usual imports. As this is an simple (synthetic)
# structured model, we can make due with few packages.

import numpy as np
import xarray as xr
import pandas as pd
import imod
import random
from matplotlib import pyplot as plt

# %%
# Create grid coordinates
# -----------------------
#
# The first steps consist of setting up the grid -- first the number of layer,
# rows, and columns. Cell sizes are constant throughout the model.

nlay = 3
nrow = 15
ncol = 15
shape = (nlay, nrow, ncol)

dx = 5000.0
dy = -5000.0
xmin = 0.0
xmax = dx * ncol
ymin = 0.0
ymax = abs(dy) * nrow
dims = ("layer", "y", "x")

layer = np.arange(nlay)
y = np.arange(ymax, ymin, dy) + 0.5 * dy
x = np.arange(xmin, xmax, dx) + 0.5 * dx
coords = {"layer": layer, "y": y, "x": x}

# %%
# Create DataArrays
# -----------------
#
# Now that we have the grid coordinates setup, we can start defining model
# parameters. The model is characterized by:
#
# * a constant head boundary on the left
# * a single drain in the center left of the model
# * uniform recharge on the top layer
# * a number of wells scattered throughout the model.

idomain = xr.DataArray(np.ones(shape, dtype=int), coords=coords, dims=dims)
bottom_lay1 = -200
thick_lay = 100 # assume all the layers have the same thickness
bottom_lays = list(range(bottom_lay1 -thick_lay*(nlay-1), bottom_lay1+1, thick_lay))
# convert to float and flip the list
bottom_lays = list(reversed([float(i) for i in bottom_lays]))
bottom = xr.DataArray(bottom_lays, {"layer": layer}, ("layer",))

# Constant head
constant_head = xr.full_like(idomain, np.nan, dtype=float).sel(layer=[1, 2])
constant_head[..., 0] = 0.0
# constant_head[..., -1] = 0.0
# constant_head[:, 0,:] = 0.0 
# constant_head[:, -1,:] = 0.0 


# Drainage
elevation = xr.full_like(idomain.sel(layer=1), np.nan, dtype=float)
conductance = xr.full_like(idomain.sel(layer=1), np.nan, dtype=float)
elevation[7, 1:10] = np.array([0.0, 0.0, 10.0, 20.0, 30.0, 50.0, 70.0, 90.0, 100.0])
conductance[7, 1:10] = 1.0
fig = plt.figure()
elevation.plot.imshow()

# Recharge
rch_rate = xr.full_like(idomain.sel(layer=1), 3.0e-8, dtype=float)

# Well
# n_wells = 15
# well_layer = [3, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
# well_row = [5, 4, 6, 9, 9, 9, 9, 11, 11, 11, 11, 13, 13, 13, 13]
# well_column = [11, 6, 12, 8, 10, 12, 14, 8, 10, 12, 14, 8, 10, 12, 14]
# well_rate = [-5.0] * 15
n_wells = 4
well_layer = [3, 2 , 1 , 1]
well_row = [5, 11 , 11, 5]
well_column = [11, 4, 7, 7]

# visualization
well_map = xr.full_like(idomain.sel(layer=0), np.nan, dtype=float)
# Set the values
for ii, jj, val in zip(well_row, well_column, well_layer):
    well_map[ii, jj] = val
fig = plt.figure()
well_map.plot.imshow()

# Node properties
#  ICELLTYPE == 0: Confined cell - Constant transmissivity 
# ICELLTYPE != 0: Convertible cell - Transmissivity varies depending on the calculated head in the cell
icelltype = xr.DataArray([1, 0, 0], {"layer": layer}, ("layer",))
# K  = np.full(shape = (nlay, nrow,ncol), fill_value = np.random.uniform(0.001, 1) )
# k = xr.DataArray(K, coords=coords, dims=dims) # Horizontal hydraulic conductivity ($m/s$)
k = xr.DataArray([1.0e-3, 1.0e-4, 2.0e-4], {"layer": layer}, ("layer",))
k33 = xr.DataArray([2.0e-8, 2.0e-8, 2.0e-8], {"layer": layer}, ("layer",))

# Time:  two stress periods, the first ending at 0.5d and the second ending at 200d :
start_date = pd.to_datetime("2020-01-01")
n_times = 5
final_time = 200
all_times = np.linspace(0, final_time, n_times+1)
duration = pd.to_timedelta([str(s) + 'd' if s!=0 else str(int(s)) for s in all_times ])
# Collect time discretization
times = start_date + duration


# well rate changing with time
# Q_well = -5.0  
# well_rate_t1 = [Q_well] * n_wells # rate for the first stress period
# well_rate_t2 = [10 *Q_well] * n_wells
min_rate = -15
max_rate = 10

well_rate_allT = []
for t in range(n_times):
       # pumping rate for the current time
       well_rate_t = [random.choice(range(min_rate, max_rate)) for _ in range(n_wells)]
       # convert to float
       well_rate_t = [float(i) for i in well_rate_t]
       well_rate_allT.append(well_rate_t)

# well_rate = xr.DataArray([well_rate_t1, well_rate_t2], coords={"time": times[:-1], "well_nr": list(range(1, n_wells+1))}, dims=("time","well_nr"))
well_rate = xr.DataArray(well_rate_allT, coords={"time": times[:-1], "well_nr": list(range(1, n_wells+1))}, dims=("time","well_nr"))


times_plot = range(n_times)
fig = plt.figure()
for i in range(len(well_rate_allT[0])):
    plt.plot(times_plot,[pt[i] for pt in well_rate_allT],label = 'id %s'%i)
plt.legend()
plt.show()


# Recharge
rch_rate_allT = []
for t in range(n_times):
       rch_rate = 3.0e-8 # Recharge rate ($m/s$)
       rch_rate_t = xr.full_like(idomain.sel(layer=1), rch_rate, dtype=float) 
       rch_rate_allT.append(rch_rate_t)

# rechanrge changing in time
rch_rate = xr.concat(rch_rate_allT, dim="time").assign_coords(time = times[:-1])



# %%
# Write the model
# ---------------
#
# The first step is to define an empty model, the parameters and boundary
# conditions are added in the form of the familiar MODFLOW packages.

gwf_model = imod.mf6.GroundwaterFlowModel()
gwf_model["dis"] = imod.mf6.StructuredDiscretization(
    top=200.0, bottom=bottom, idomain=idomain
)
gwf_model["chd"] = imod.mf6.ConstantHead(
    constant_head, print_input=True, print_flows=True, save_flows=True
)
gwf_model["drn"] = imod.mf6.Drainage(
    elevation=elevation,
    conductance=conductance,
    print_input=True,
    print_flows=True,
    save_flows=True,
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
    specific_storage=1.0e-5,
    specific_yield=0.15,
    transient=False,
    convertible=0,
)
gwf_model["oc"] = imod.mf6.OutputControl(save_head="all", save_budget="all")
gwf_model["rch"] = imod.mf6.Recharge(rch_rate)
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
# Collect time discretization
simulation.create_time_discretization(additional_times=times)

# %%
# We'll create a new directory in which we will write and run the model.

modeldir = imod.util.temporary_directory()
simulation.write(modeldir)

# %%
# Run the model
# -------------
#
# .. note::
#
#   The following lines assume the ``mf6`` executable is available on your PATH.
#   :ref:`The Modflow 6 examples introduction <mf6-introduction>` shortly
#   describes how to add it to yours.

simulation.run()

# %%
# Open the results
# ----------------
#
# We'll open the heads (.hds) file.

head = imod.mf6.open_hds(
    modeldir / "GWF_1/GWF_1.hds",
    modeldir / "GWF_1/dis.dis.grb",
)

# %%
# Visualize the results
# ---------------------
# head.isel(layer=0, x=5, y=5).plot()

for i in range(n_times-1):
        fig = plt.figure()
        head.isel(layer=1, time=i).plot.contourf()
# %%
# .. _MODFLOW6 example problems: https://github.com/MODFLOW-USGS/modflow6-examples
# .. _description: https://modflow6-examples.readthedocs.io/en/master/_examples/ex-gwf-twri.html
# .. _notebook: https://github.com/MODFLOW-USGS/modflow6-examples/tree/master/notebooks/ex-gwf-twri.ipynb
# .. _Techniques of Water-Resources Investigation: https://pubs.usgs.gov/twri/twri7-c1/
# .. _McDonald & Harbaugh, 1988: https://pubs.er.usgs.gov/publication/twri06A1
# .. _Harbaugh & McDonald, 1996: https://pubs.er.usgs.gov/publication/ofr96485
# .. _Harbaugh, 2005: https://pubs.er.usgs.gov/publication/tm6A16
# .. _FloPy: https://github.com/modflowpy/flopy

# %%
