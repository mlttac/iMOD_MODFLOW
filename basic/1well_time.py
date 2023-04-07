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

dx = 500.0
dy = -500.0
zmin = -100.0
xmin = 0.0
xmax = dx * ncol
ymin = 0.0
ymax = abs(dy) * nrow
dims = ("layer", "y", "x")

layer = np.array([1]) # this is different in case of more layers!
y = np.arange(ymax, ymin, dy) + 0.5 * dy
x = np.arange(xmin, xmax, dx) + 0.5 * dx
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
n_wells = 4
well_layer = [1, 1 , 1 , 1]
well_row = [5, 15 , 11, 5]
well_column = [11, 15, 7, 7]

# n_wells = 1
# well_layer = [1]
# well_row = [15 ]
# well_column = [15]


# Node properties
icelltype = xr.DataArray([1], {"layer": layer}, ("layer",))
k = xr.DataArray([1.0e-3], {"layer": layer}, ("layer",))
k33 = xr.DataArray([2.0e-3], {"layer": layer}, ("layer",))

start_date = pd.to_datetime("2020-01-01")
n_times = 3
final_time = 200
all_times = np.linspace(0, final_time, n_times+1)
duration = pd.to_timedelta([str(s) + 'd' if s!=0 else str(int(s)) for s in all_times ])
times = start_date + duration

# well rate changing with time
min_rate = -3
max_rate = 10

well_rate_allT = []
for t in range(n_times):
       # pumping rate for the current time
       well_rate_t = [random.choice(range(min_rate, max_rate)) for _ in range(n_wells)]
       # convert to float
       well_rate_t = [float(i)/100 for i in well_rate_t]
       well_rate_allT.append(well_rate_t)

well_rate = xr.DataArray(well_rate_allT, coords={"time": times[:-1], "well_nr": list(range(1, n_wells+1))}, dims=("time","well_nr"))



transient = xr.DataArray(
    [False, True, True, True], {"time": times}, ("time",)
)


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
    specific_storage=1.0e-5,
    specific_yield=0.15,
    transient=transient,
    convertible=0,
)
gwf_model["oc"] = imod.mf6.OutputControl(save_head="all", save_budget="all")
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

modeldir = imod.util.temporary_directory()
simulation.write(modeldir)
simulation.run()

# Open the results
head = imod.mf6.open_hds(
    modeldir / "GWF_1/GWF_1.hds",
    modeldir / "GWF_1/dis.dis.grb",
)

# Visualize the results
# ---------------------
# head.isel(layer=0, x=5, y=5).plot()

for i in range(n_times-1):
        fig = plt.figure()
        head.isel(layer=0, time=i).plot.contourf()
