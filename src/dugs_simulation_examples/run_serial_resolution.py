import os

import numpy as np
import pandas as pd
from geomodel.generate_models import stratified_model_simulation


def execute(proxy_model, file_name, output_dir):
    # now we start to run for the time report--------------------------------------------------------------
    time_step = proxy_model.report_time
    even_end = int(proxy_model.total_time / time_step) * time_step
    time_step_arr = np.ones(int(proxy_model.total_time / time_step)) * time_step
    if proxy_model.runtime - even_end > 0:
        time_step_arr = np.append(time_step_arr, proxy_model.total_time - even_end)
    time_index = 0
    # proxy_model.output_to_vtk(ith_step=time_index, output_directory='vtk')
    properties = proxy_model.output_properties()
    pressure = properties[0, :]
    temperature = properties[2, :]
    proxy_model.xrdata.set_dynamic_grid_xarray(
        time_index,
        "Pressure",
        np.rot90(
            pressure.reshape(
                (proxy_model.nx, proxy_model.ny, proxy_model.nz), order="F"
            )
        ),
    )
    proxy_model.xrdata.set_dynamic_grid_xarray(
        time_index,
        "Temperature",
        np.rot90(
            temperature.reshape(
                (proxy_model.nx, proxy_model.ny, proxy_model.nz), order="F"
            )
        ),
    )
    for ts in time_step_arr:
        for _, w in enumerate(proxy_model.reservoir.wells):
            if w.name.lower().startswith("i"):
                # w.control = proxy_model.physics.new_rate_water_inj(7500, 300)
                w.control = proxy_model.physics.new_mass_rate_water_inj(417000, 2358)
                w.constraint = proxy_model.physics.new_bhp_water_inj(1858 * 0.15, 300)
                # w.constraint = self.physics.new_bhp_water_inj(200, self.inj_temperature)
            else:
                # w.control = proxy_model.physics.new_rate_water_prod(7500)
                w.control = proxy_model.physics.new_mass_rate_water_prod(417000)
                # w.constraint = proxy_model.physics.new_bhp_prod(183)

        proxy_model.run(ts)
        # fluxes = np.array(proxy_model.physics.engine.fluxes, copy=False)
        # vels = proxy_model.reconstruct_velocities(fluxes[proxy_model.physics.engine.P_VAR::proxy_model.physics.N_VAR])
        time_index += 1
        properties = proxy_model.output_properties()
        pressure = properties[0, :]
        temperature = properties[2, :]
        proxy_model.xrdata.set_dynamic_grid_xarray(
            time_index,
            "Pressure",
            np.rot90(
                pressure.reshape(
                    (proxy_model.nx, proxy_model.ny, proxy_model.nz), order="F"
                )
            ),
        )
        proxy_model.xrdata.set_dynamic_grid_xarray(
            time_index,
            "Temperature",
            np.rot90(
                temperature.reshape(
                    (proxy_model.nx, proxy_model.ny, proxy_model.nz), order="F"
                )
            ),
        )
        proxy_model.physics.engine.report()
    proxy_model.print_timers()
    proxy_model.print_stat()

    time_data_report = pd.DataFrame.from_dict(
        proxy_model.physics.engine.time_data_report
    )
    proxy_model.xrdata.set_dynamic_xarray(proxy_model.reservoir.wells, time_data_report)

    proxy_model.xrdata.write_xarray(file_name=file_name, output_dir=output_dir)

    return time_data_report


def run_simulation():
    """Give the input of different nx, ny and nz to proxy_model_simulation

    :return:
    """
    # list_dx = [81, 27, 18, 9, 6, 3]
    list_dz = np.array([15, 8, 5, 4])
    dx = 30
    # dz = 10
    output_directory = f"serial_resolution_3_same_layered_dz_test"
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    for i in list_dz:
        print("\n")
        print(f"dx {dx:.2f}, dy {dx:.2f}, " f"dz {i:.2f}")
        print("\n")
        # geothermal_model = heterogeneous_model_simulation(int(x_spacing / dx),
        #                                                           int(y_spacing / dx),
        #                                                           int(z_spacing / i),
        #                                                           dx,
        #                                                           dx,
        #                                                           i,
        #                                                          )
        # geothermal_model = homogeneous_model_simulation(int(x_spacing/i),
        #                                                        int(y_spacing/i),
        #                                                        int(z_spacing/dz),
        #                                                        i,
        #                                                        i,
        #                                                        dz,
        #                                                        file_name=f'output_{i}',
        #                                                        output_dir=output_directory)
        geothermal_model = stratified_model_simulation(
            int(x_spacing / dx), int(y_spacing / dx), int(z_spacing / i), dx, dx, i
        )
        temperature = execute(
            geothermal_model, file_name=f"output_{i}", output_dir=output_directory
        )

        output_path = os.path.relpath(
            f"{output_directory}/temperature_resolution_dz.csv"
        )
        if os.path.exists(output_path):
            df = pd.read_csv(output_path, delimiter=",")
            df[f"{i:.2f}"] = temperature["PRD1 : temperature (K)"]
            df.to_csv(output_path, index=False)
        else:
            temperature.rename(
                columns={"PRD1 : temperature (K)": f"{i:.2f}"}, inplace=True
            )
            temperature[["time", f"{i:.2f}"]].to_csv(output_path, index=False)


if __name__ == "__main__":
    x_spacing = 4500
    y_spacing = 4200
    z_spacing = 100
    run_simulation()
