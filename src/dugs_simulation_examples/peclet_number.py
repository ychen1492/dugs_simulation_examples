import os

import numpy as np
from darts.discretizer import value_vector
from darts.physics.properties.iapws.iapws_property import iapws_total_enthalpy_evalutor, iapws_water_density_evaluator
from matplotlib import pyplot as plt
from src.model import XarrayApi

plt.rcParams["font.family"] = "DejaVu Sans"
plotparams = {
    "axes.titlesize": "12",
    "axes.labelsize": "12",
    "axes.spines.left": False,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.spines.bottom": False,
    "axes.grid": True,
    "xtick.labelsize": "14",
    "ytick.labelsize": "14",
    "figure.titlesize": "20",
    "figure.titleweight": "bold",
    "grid.linestyle": "-",
    "grid.linewidth": 0.5,
    "grid.alpha": "0.5",
    "ytick.major.width": 0,
    "xtick.major.width": 0,
    "axes.unicode_minus": False,
    "legend.fontsize": 9,
    #   "text.usetex": True,
    "xtick.bottom": False,
    "xtick.major.top": False,
    "xtick.minor.visible": True,
    # 'xtick.major.bottom':True,
    # 'xtick.labelbottom': True,
    "ytick.major.left": True,
    "ytick.labelright": False,
}


plt.rcParams.update(plotparams)

# Basic parameters
dx = dy = 18  # grid spacing in x and y, m
dz = 6  # grid spacing in z, m
dt = 365  # time step, days
number_of_time_steps = 30
discharge_rate = 7500  # m3/day

density_evaluator = iapws_water_density_evaluator()

################################# Conduction #########################################
# Thermal conductivity of the rock and water
kappa_sandstone = 3 * 86.4  # sandstone, kJ/m/day/K
kappa_shale = 2.2 * 86.4  # shale, kJ/m/day/K
kappa_water = 0.6 * 86.4  # water, kJ/m/day/K

models = ["ho", "str", "he"]
directory = r"C:\Users\ychen62\Repository\reference-simulation\base"
ncols = 3
fig, ax = plt.subplots(ncols=ncols, nrows=1, figsize=(15, 7))
for i, category in enumerate(models):
    peclet = []
    dimensionless_time = []
    output = XarrayApi.get_xarray(
        file_name=os.path.join(directory, f"base_30_steps_{category}.nc")
    )
    for index in range(number_of_time_steps):
        temperature = output["Temperature"].isel(time=index + 1).data
        production_temperature = output["Production Well 1 Temperature"].data[index + 1]
        production_well_bhp = output["Production Well 1 Pressure"].data[index + 1]
        kappa_rock = np.zeros_like(output["Temperature"].isel(time=index + 1).data)
        kappa_rock[output["Poro"] <= 0.1] = kappa_shale
        kappa_rock[output["Poro"] > 0.1] = kappa_sandstone
        # Calculate the gradient of T
        grad_T = np.gradient(temperature, dy, dx, dz, axis=(0, 1, 2))

        # Calculate the heat flux at this time step
        heat_flux_rock = kappa_rock * np.array(grad_T)
        heat_flux_water = kappa_water * np.array(grad_T)
        # *
        heat_flux = heat_flux_rock + heat_flux_water

        # Calculate the divergence of the heat flux at this time step
        div_heat_flux = (
            np.gradient(heat_flux[0], dy, axis=0)
            + np.gradient(heat_flux[1], dx, axis=1)
            + np.gradient(heat_flux[2], dz, axis=2)
        )

        # Integrate over the spatial domain for each time step
        spatial_integral = np.sum(div_heat_flux, axis=(0, 1, 2)) * (dx * dy * dz)

        # Total conduction at this time step
        total_conduction = np.abs(spatial_integral) * dt
        ################################# Convection #########################################
        # Convection only happens at reservoir layers
        all_temperature = np.transpose(
            output["Temperature"].isel(time=index + 1).data[:, :, 4:20], (2, 0, 1)
        )
        all_pressure = np.transpose(
            output["Pressure"].isel(time=index + 1).data[:, :, 4:20], (2, 0, 1)
        )

        flatten_temperature = np.mean(all_temperature.flatten())
        flatten_pressure = np.mean(all_pressure.flatten())

        state = value_vector([production_well_bhp, 0])
        E = iapws_total_enthalpy_evalutor(production_temperature)
        enthalpy = E.evaluate(state) / 18.015

        water_density = density_evaluator.evaluate(
            [production_well_bhp, enthalpy * 18.015]
        )

        # Total convection at this time step
        total_convection = water_density * enthalpy * discharge_rate * dt
        peclet.append(total_convection / total_conduction)
        dimensionless_time.append(
            discharge_rate
            * dt
            * (index + 1)
            / np.sum(output["Poro"].data[4:20, :, :] * (dx * dy * dz), axis=(0, 1, 2))
        )

    ax[0].plot(dimensionless_time, peclet, label=category)
    ax[1].plot(output["Production Well 1 Temperature"][1:], label=category)
    ax[2].plot(-1 * output["Production Well 1 Energy"][1:], label=category)

for n in range(1, ncols):
    ax[n].set_xscale("log")
    ax[n].legend()
    ax[n].set_xlabel("Time [years]")
ax[0].legend()

ax[0].set_xlabel("Dimensionless time [-]")
ax[0].set_xscale("log")
ax[0].set_ylabel("Peclet number [-]")
ax[1].set_ylabel("Temperature [K]")
plt.tight_layout()
ax[2].set_ylabel("Power [kJ/day]")
# plt.savefig(fname='temp_1.png', dpi=300, bbox_inches='tight', pad_inches=0.5)
plt.show()
