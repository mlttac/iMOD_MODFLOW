"""
# .. _MODFLOW6 example problems: https://github.com/MODFLOW-USGS/modflow6-examples
# .. _description: https://modflow6-examples.readthedocs.io/en/master/_examples/ex-gwf-twri.html
# .. _notebook: https://github.com/MODFLOW-USGS/modflow6-examples/tree/master/notebooks/ex-gwf-twri.ipynb
# .. _iMOD Python: https://deltares.gitlab.io/imod/imod-python/index.html

"""

# ASK JOERI


# Results: 1) where saved, 2)how to save in arrays, 3)time dependent results
# Transient results
# Calculate water balance


# %%
# We'll start with the usual imports. As this is an simple (synthetic)
# structured model, we can make due with few packages.

import numpy as np
import xarray as xr
import pandas as pd
import imod

# %%
# Create grid coordinates

nlay = 1 # Number of layers
nrow = 32 # Number of columns
ncol = 32 # Number of columns
dx = 3000 # Column width ($m$)
dy = -3000  # Row width ($m$)
zmin = -150.0 # Layer bottom elevations ($m$)
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
Q_well = -0.5  #m2/day


# %%
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
constant_head.sel(layer=1).plot.imshow()

# Drainage
elevation = xr.full_like(idomain.sel(layer=1), np.nan, dtype=float)
conductance = xr.full_like(idomain.sel(layer=1), np.nan, dtype=float)

n_drains = 10
y_drain = 15 # start from the top
x_drain = 20 # start from the left
z_drain_top = -10 
dz_lowerDrains = np.arange(0, n_drains+1)
cond_drain = 1

elevation[y_drain,x_drain] = z_drain_top
# elevation.plot.imshow()
conductance[y_drain,x_drain] = cond_drain
drain_active = True

# Time:  two stress periods, the first ending at 0.5d and the second ending at 200d :
start_date = pd.to_datetime("2020-01-01")
duration = pd.to_timedelta(["0", "0.5d", "200d"])
# Collect time discretization
times = start_date + duration

wells_active = False
# Well
well_layer = [1, 1]
well_row = [15, 15]
well_column = [3, 22]
n_wells = len(well_column)

#fix well rate in time
# well_rate = [Q_well] * n_wells

# well rate changing with time
well_rate_t1 = [Q_well] * n_wells # rate for the first stress period
well_rate_t2 = [10 *Q_well] * n_wells
well_rate = xr.DataArray([well_rate_t1, well_rate_t2], coords={"time": times[:-1], "well_nr": [1, 2]}, dims=("time","well_nr"))


# Recharge
rch_rate_1 = xr.full_like(idomain.sel(layer=1), 10.0e-8, dtype=float) # Recharge rate ($m/s$)
rch_rate_2 = xr.full_like(idomain.sel(layer=1), 10.0e-6, dtype=float) # Recharge rate ($m/s$)
# rechanrge changing in time
rch_rate = xr.concat([rch_rate_1, rch_rate_2], dim="time").assign_coords(time = times[:-1])

# Node properties
icelltype = xr.DataArray([1], {"layer": layer}, ("layer",))
# k  = np.full(shape = (nrow,ncol), fill_value = np.random.uniform(5, 25) )

K  = np.full(shape = (nlay, nrow,ncol), fill_value = np.random.uniform(0.001, 1) )
k = xr.DataArray(K, coords=coords, dims=dims) # Horizontal hydraulic conductivity ($m/s$)
k33 = k # Vertical hydraulic conductivity ($m/day$)

# %% Write the model
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
                  elevation=elevation - dz,
                  conductance=conductance,
                  print_input=True,
                  print_flows=True,
                  save_flows=True,
              )
              
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

# %%
# We'll create a new directory in which we will write and run the model.


modeldir = imod.util.temporary_directory()

# if I want to store the results somewhere
# from pathlib import Path
# modeldir = Path('C:/Users/cnmlt/AppData/Local/imod-python')

simulation.write(modeldir)

# %%
# Run the model

simulation.run( )

# %%
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

# %%
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
head.isel(layer=0, x=x_drain, y=y_drain)

# to store the results I can actually use the xarrays functionalities:
# example: xr.concat([head, head], dim="scenario")


