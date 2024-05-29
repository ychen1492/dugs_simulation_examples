import os

import numpy as np
import pandas as pd
from darts.engines import redirect_darts_output
from model import Model
from reference_simulation_utils.read_files import from_las_to_poro_gamma, read_pickle_file
from reference_simulation_utils.xarray_api import XarrayApi

# report_time = 10
# total_time = 1000
# perm = 374.9389
# poro = 0.1578
#
# x_spacing = 240
# y_spacing = 160
# z_spacing = 42
# set_dx = 1
# set_nx = int(x_spacing / set_dx)
# set_dy = 1
# set_ny = int(y_spacing / set_dy)
# set_dz = 1
# set_nz = int(z_spacing / set_dz)
# overburden = 25
# underburden = 25

report_time = 365
total_time = 365 * 30
perm = 3000
poro = 0.2

x_spacing = 4500
y_spacing = 4200
z_spacing = 100
set_dx = 15
set_nx = int(x_spacing / set_dx)
set_dy = 70
set_ny = int(y_spacing / set_dy)
set_dz = 8
set_nz = int(z_spacing / set_dz)
overburden = 25
underburden = 25


def generate_base_ho():
    """This is a function without any inputs to generate vtk and time data file for homogeneous case

    :return:
        vtk and time data excel file
    """
    redirect_darts_output("log.txt")
    perms = np.ones(set_nx * set_ny * set_nz) * perm
    poros = np.ones(set_nx * set_ny * set_nz) * poro
    xrdata = XarrayApi(
        nx=set_nx,
        ny=set_ny,
        nz=set_nz + overburden + underburden,
        total_time=total_time,
        report_time=report_time,
        number_of_inj_wells=1,
        number_of_prd_wells=1,
    )
    proxy_model = Model(
        total_time=total_time,
        set_nx=set_nx,
        set_ny=set_ny,
        set_nz=set_nz,
        set_dx=set_dx,
        set_dy=set_dy,
        set_dz=set_dz,
        perms=perms,
        poro=poros,
        report_time_step=report_time,
        overburden=overburden,
        xrdata=xrdata,
    )
    if not os.path.exists("RealBaseTest1"):
        os.mkdir("RealBaseTest1")
    proxy_model.init()
    output_path_nc = os.path.join("RealBaseTest1", "data_new_without_larger.nc")
    proxy_model.run(export_to_vtk=False, output_path=output_path_nc)

    proxy_model_elapsed_time = (
        proxy_model.timer.node["initialization"].get_timer()
        + proxy_model.timer.node["simulation"].get_timer()
    )

    td = pd.DataFrame.from_dict(proxy_model.physics.engine.time_data_report)

    output_path = os.path.relpath(f"./RealBaseTest1/base_resolution_ho.xlsx")
    writer = pd.ExcelWriter(output_path)
    td.to_excel(writer, "Sheet1")
    writer.close()
    with open("./RealBaseTest1/simulation_time_resolution_ho.txt", "w") as f:
        f.write(f"{proxy_model_elapsed_time}")


def generate_base_stratified():
    """This is a function without any inputs to generate vtk and time data file for stratified case

    :return:
        vtk and time data excel file
    """
    # read porosity from the file
    org_poro = from_las_to_poro_gamma(
        "LogData/Well_PIJNACKER_GT_01_depth_gamma_4.las", set_nz
    )
    poro = np.concatenate([np.ones(set_nx * set_ny) * p for p in org_poro], axis=0)
    # calculate permeability, this is from Duncan's thesis
    org_perm = np.exp(
        110.744 * poro**3 - 171.8268 * poro**2 + 92.9227 * poro - 2.047
    )
    perms = org_perm
    redirect_darts_output("log.txt")
    proxy_model = Model(
        total_time=total_time,
        set_nx=set_nx,
        set_ny=set_ny,
        set_nz=set_nz,
        set_dx=set_dx,
        set_dy=set_dy,
        set_dz=set_dz,
        perms=perms,
        poro=poro,
        report_time_step=report_time,
        overburden=overburden,
    )
    proxy_model.init()
    proxy_model.run(export_to_vtk=True)

    proxy_model_elapsed_time = (
        proxy_model.timer.node["initialization"].get_timer()
        + proxy_model.timer.node["simulation"].get_timer()
    )

    td = pd.DataFrame.from_dict(proxy_model.physics.engine.time_data_report)

    if not os.path.exists("RealBase"):
        os.mkdir("RealBase")
    output_path = os.path.relpath(f"./RealBase/base_resolution_layered.xlsx")
    writer = pd.ExcelWriter(output_path)
    td.to_excel(writer, "Sheet1")
    writer.close()
    with open("./RealBase/simulation_time_resolution_layered.txt", "w") as f:
        f.write(f"{proxy_model_elapsed_time}")


def generate_base_he():
    """This is a function without any inputs to generate vtk and time data file for heterogeneous case

    :return:
        vtk and time data excel file
    """
    poros, perms = read_pickle_file(set_ny, set_nx, "Porosity")
    redirect_darts_output("log.txt")
    proxy_model = Model(
        total_time=total_time,
        set_nx=set_nx,
        set_ny=set_ny,
        set_nz=set_nz,
        set_dx=set_dx,
        set_dy=set_dy,
        set_dz=set_dz,
        perms=perms,
        poro=poros,
        report_time_step=report_time,
        overburden=overburden,
    )
    proxy_model.init()
    proxy_model.run(export_to_vtk=True)

    proxy_model_elapsed_time = (
        proxy_model.timer.node["initialization"].get_timer()
        + proxy_model.timer.node["simulation"].get_timer()
    )

    td = pd.DataFrame.from_dict(proxy_model.physics.engine.time_data_report)

    if not os.path.exists("RealBase"):
        os.mkdir("RealBase")
    output_path = os.path.relpath(f"./RealBase/base_resolution_he.xlsx")
    writer = pd.ExcelWriter(output_path)
    td.to_excel(writer, "Sheet1")
    writer.close()
    with open("./RealBase/simulation_time_resolution_he.txt", "w") as f:
        f.write(f"{proxy_model_elapsed_time}")


if __name__ == "__main__":
    generate_base_ho()
