from geomodel.generate_models import heterogeneous_model_simulation, homogeneous_model_simulation, stratified_model_simulation
from run_serial_resolution import execute


def generate_base():
    geothermal_model = homogeneous_model_simulation(
        nx, ny, nz, dx, dy, dz, overburden=4
    )
    geothermal_model = stratified_model_simulation(nx, ny, nz, dx, dy, dz, overburden=4)
    geothermal_model = heterogeneous_model_simulation(
        nx, ny, nz, dx, dy, dz, overburden=4
    )
    _ = execute(geothermal_model, file_name="base_30_steps_he_0", output_dir="base")


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
