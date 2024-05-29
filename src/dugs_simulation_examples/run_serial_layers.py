import os

import numpy as np
import pandas as pd
from generate_models import homogeneous_model_simulation


def execute_output_temperature(simulation_model):
    # now we start to run for the time report--------------------------------------------------------------
    time_step = simulation_model.report_time
    even_end = int(simulation_model.total_time / time_step) * time_step
    time_step_arr = np.ones(int(simulation_model.total_time / time_step)) * time_step
    if simulation_model.runtime - even_end > 0:
        time_step_arr = np.append(time_step_arr, simulation_model.total_time - even_end)

    for ts in time_step_arr:
        for _, w in enumerate(simulation_model.reservoir.wells):
            if w.name.lower().startswith("i"):
                w.control = simulation_model.physics.new_rate_water_inj(7500, 300)
                # w.control = simulation_model.physics.new_mass_rate_water_inj(417000, 1914.13)
                # w.constraint = self.physics.new_bhp_water_inj(200, self.inj_temperature)
            else:
                w.control = simulation_model.physics.new_rate_water_prod(7500)
                # w.control = simulation_model.physics.new_mass_rate_water_inj(417000, 1914.13)

        simulation_model.run(ts)
        simulation_model.physics.engine.report()

    properties = simulation_model.output_properties()
    temperature = properties[2, :]

    return temperature


def run_simulation():
    overburden_layers = 2
    output_directory = "serial_layers_ho"
    file_name = "layers_info.csv"
    # thickness includes both overburden and underburden layers
    layers_dict = {"number_of_layers": [], "thickness": []}
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    # write the number of overburden layers to file
    layers_dict["number_of_layers"].append(overburden_layers)
    layers_dict["thickness"].append(overburden_layers * dz * 2)

    print("\n")
    print(f"overburden layers: {overburden_layers}")
    print("\n")
    geothermal_model = homogeneous_model_simulation(
        nx, ny, nz + 2 * overburden_layers, dx, dy, dz, overburden=overburden_layers
    )
    temperature = execute_output_temperature(geothermal_model)

    # the temperature distribution of the first layer
    top_layer_temp = temperature.reshape(nx, ny, nz + 2 * overburden_layers, order="F")[
        :, :, 0
    ].flatten(order="F")

    while np.abs(min(top_layer_temp) - max(top_layer_temp)) > 0.05:
        overburden_layers += 2
        layers_dict["number_of_layers"].append(overburden_layers)
        layers_dict["thickness"].append(overburden_layers * dz * 2)
        print("\n")
        print(f"overburden layers: {overburden_layers}")
        print("\n")
        temperature = homogeneous_model_simulation(
            nx, ny, nz + 2 * overburden_layers, dx, dy, dz, overburden=overburden_layers
        )

        top_layer_temp = temperature.reshape(
            nx, ny, nz + 2 * overburden_layers, order="F"
        )[:, :, 0].flatten(order="F")

    output_path = os.path.relpath(f"{output_directory}/{file_name}")
    output_df = pd.DataFrame.from_dict(layers_dict)
    output_df.to_csv(output_path, index=False)
    print("\n")
    print(f"The minimum number of confining layers is: {overburden_layers}")


if __name__ == "__main__":
    x_spacing = 4500
    y_spacing = 4200
    z_spacing = 100
    dx = dy = 18
    dz = 6
    nx = int(x_spacing / dx)
    ny = int(y_spacing / dy)
    nz = int(z_spacing / dz)
    run_simulation()
