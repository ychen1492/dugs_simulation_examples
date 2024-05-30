import os
import pickle

import numpy as np
import pyvista as pv
from darts.engines import redirect_darts_output

from .model import Model


def output_grid(nx, ny, permeability, nz=12):
    """Generate 3D models using the pyvista for
    the given dimensions and write to the file.
    By default, the length in x, y and z are set.

    :param nx: number of the grid cells in x direction
    :type nx: int
    :param ny: number of the grid cells in y direction
    :type ny: int
    :param permeability: the permeability distribution of the given model
    :type permeability: np.ndarray
    :param nz: _number of the grid cells in z direction, defaults to 12
    :type nz: int, optional
    """
    x_spacing = 4500
    y_spacing = 4200
    z_spacing = 100
    shape = (nx, ny, nz)

    spacing = (x_spacing / shape[0], y_spacing / shape[1], z_spacing / shape[2])
    origin = (spacing[0] / 2.0, spacing[1] / 2.0, -2300)
    x = np.linspace(
        origin[0] - spacing[0] / 2.0,
        origin[0] + spacing[0] * (shape[0] - 0.5),
        shape[0] + 1,
    )
    y = np.linspace(
        origin[1] - spacing[1] / 2.0,
        origin[1] + spacing[1] * (shape[1] - 0.5),
        shape[1] + 1,
    )
    x, y = np.meshgrid(x, y, indexing="ij")
    z = np.linspace(
        np.full(x.shape, origin[2] - spacing[2] / 2.0),
        np.full(x.shape, origin[2] + spacing[2] * (shape[2] - 0.5)),
        num=shape[2] + 1,
        axis=-1,
    )
    x = np.repeat(x[..., np.newaxis], shape[2] + 1, axis=-1)
    y = np.repeat(y[..., np.newaxis], shape[2] + 1, axis=-1)

    grid = pv.StructuredGrid(x, y, z)

    line_1 = pv.Line((2400, 2100, -46583), (2400, 2100, -40783))
    line_2 = pv.Line((3700, 2100, -46583), (3700, 2100, -40783))
    tube_radius = 30
    tube_1 = line_1.tube(radius=tube_radius)
    tube_2 = line_2.tube(radius=tube_radius)

    grid.cell_data["Permeability [mD]"] = permeability

    p = pv.Plotter(off_screen=True, window_size=(800, 900))
    # p = pv.Plotter(window_size=(800, 900))

    p.add_mesh(
        grid.scale([1.0, 1.0, 20.0], inplace=False),
        scalars="Permeability [mD]",
        log_scale=True,
    )
    p.add_mesh(tube_1, color="blue")
    p.add_mesh(tube_2, color="red")

    p.show_axes()
    p.show()
    p.screenshot(
        r"U:\YuanGeothermalSimulation\reference-simulation\
            Journal\20240421_heterogeneous_3d_model.png"
    )


def generate_poro_normalized_distribution(task_id, size=50):
    # Define mean and standard deviation
    mean = 0.15
    std_dev = 0.05

    np.random.seed(int(task_id))
    while True:
        porosity = np.random.normal(mean, std_dev, size)
        porosity = porosity[(porosity >= 0.04) & (porosity <= 0.28)]
        if len(porosity) > 0:
            break

    return porosity


def upscale_porosity(poro, nz):
    # Use linear interpolation to upscale the array
    upscaled_array = np.interp(
        np.linspace(0, 1, nz), np.linspace(0, 1, poro.size), poro
    )

    return upscaled_array


def stratified_model_simulation(nx, ny, nz, dx, dy, dz, n_points=32, overburden=0):
    org_poro = generate_poro_normalized_distribution(100)
    po = upscale_porosity(org_poro, nz)

    poros = np.concatenate([np.ones(nx * ny) * p for p in po], axis=0)

    org_perm = np.array(
        [
            pow(10, x)
            for x in (-3.523e-7) * (poros * 100) ** 5
            + 4.278e-5 * (poros * 100) ** 4
            - 1.723e-3 * (poros * 100) ** 3
            + 1.896e-2 * (poros * 100) ** 2
            + 0.333 * (poros * 100)
            - 3.222
        ]
    )
    perms = org_perm

    proxy_model = Model(
        set_nx=nx,
        set_ny=ny,
        set_nz=nz,
        set_dx=dx,
        set_dy=dy,
        set_dz=dz,
        perms=perms,
        poro=poros,
        overburden=overburden,
        n_points=n_points,
    )
    redirect_darts_output(f"log_str_{n_points}.txt")
    proxy_model.init()
    # proxy_model.reservoir.mesh.init_grav_coef(0.)

    return proxy_model


def heterogeneous_model_simulation(nx, ny, nz, dx, dy, dz, n_points=32, overburden=0):
    redirect_darts_output(f"log_he_{n_points}.txt")

    cur_path = os.path.dirname(__file__)
    path_to_pickle = os.path.join(cur_path, "..", "..", "poro_he_new.pkl")
    perms = lambda poro: (
        (-3.523e-7) * (poro * 100) ** 5
        + 4.278e-5 * (poro * 100) ** 4
        - 1.723e-3 * (poro * 100) ** 3
        + 1.896e-2 * (poro * 100) ** 2
        + 0.333 * (poro * 100)
        - 3.222
    )
    with open(path_to_pickle, "rb") as por:
        org_porosity = pickle.load(file=por)

    shape = (500, 700, 50)

    spacing = (4500 / shape[0], 4200 / shape[1], 100 / shape[2])
    origin = (spacing[0] / 2.0, spacing[1] / 2.0, -2300)
    x = np.linspace(
        origin[0] - spacing[0] / 2.0,
        origin[0] + spacing[0] * (shape[0] - 0.5),
        shape[0] + 1,
    )
    y = np.linspace(
        origin[1] - spacing[1] / 2.0,
        origin[1] + spacing[1] * (shape[1] - 0.5),
        shape[1] + 1,
    )
    x, y = np.meshgrid(x, y, indexing="ij")
    z = np.linspace(
        np.full(x.shape, origin[2] - spacing[2] / 2.0),
        np.full(x.shape, origin[2] + spacing[2] * (shape[2] - 0.5)),
        num=shape[2] + 1,
        axis=-1,
    )
    x = np.repeat(x[..., np.newaxis], shape[2] + 1, axis=-1)
    y = np.repeat(y[..., np.newaxis], shape[2] + 1, axis=-1)

    grid = pv.StructuredGrid(x, y, z)

    grid.cell_data["Base_Porosity"] = org_porosity

    mid = pv.create_grid(grid, (nx, ny, nz)).sample(grid)
    porosity = mid.point_data["Base_Porosity"]
    porosity[porosity == 0] = org_porosity.min()
    permeability = np.power(10, perms(porosity))

    poro, perm = porosity, permeability

    proxy_model = Model(
        set_nx=nx,
        set_ny=ny,
        set_nz=nz,
        set_dx=dx,
        set_dy=dy,
        set_dz=dz,
        perms=perm,
        poro=poro,
        overburden=overburden,
        n_points=n_points,
    )
    proxy_model.init()
    # proxy_model.reservoir.mesh.init_grav_coef(0.)

    return proxy_model


def homogeneous_model_simulation(nx, ny, nz, dx, dy, dz, n_points=32, overburden=0):
    redirect_darts_output(f"log_ho_{n_points}.txt")
    perms = np.ones(nx * ny * nz) * 800
    poros = np.ones(nx * ny * nz) * 0.2

    proxy_model = Model(
        set_nx=nx,
        set_ny=ny,
        set_nz=nz,
        set_dx=dx,
        set_dy=dy,
        set_dz=dz,
        perms=perms,
        poro=poros,
        overburden=overburden,
        n_points=n_points,
    )
    proxy_model.init()
    # proxy_model.reservoir.mesh.init_grav_coef(0.)

    return proxy_model
