import numpy as np
from generate_models import stratified_model_simulation
from run_serial_resolution import execute


def generate_base():
    num_terms = 8
    obl_point = np.array([int(2**i) for i in range(1, num_terms)])
    for i in obl_point:
        print("\n")
        print(f"number of obl points: {i}")
        print("\n")
        # geothermal_model = homogeneous_model_simulation(nx, ny, nz, dx, dy, dz, overburden=4,n_points=i)
        geothermal_model = stratified_model_simulation(
            nx, ny, nz, dx, dy, dz, overburden=4, n_points=i
        )
        # geothermal_model = heterogeneous_model_simulation(nx, ny, nz, dx, dy, dz,overburden=4,n_points=i)
        _ = execute(
            geothermal_model,
            file_name=f"stratified_obl_{i}",
            output_dir="output_nc_base_obl",
        )


if __name__ == "__main__":
    x_spacing = 4500
    y_spacing = 4200
    z_spacing = 100
    dx = dy = 18
    dz = 6
    nx = int(x_spacing / dx)
    ny = int(y_spacing / dy)
    nz = int(z_spacing / dz)
    generate_base()
