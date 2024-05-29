import os

import numpy as np
import xarray as xr
from darts.discretizer import value_vector
from darts.engines import property_evaluator_iface, sim_params
from darts.models.darts_model import DartsModel
from darts.physics.geothermal.physics import Geothermal
from darts.physics.geothermal.property_container import PropertyContainer
from darts.physics.properties.iapws.iapws_property import iapws_water_density_evaluator, temperature_region1_evaluator
from darts.reservoirs.cpg_reservoir import CPG_Reservoir
from darts.reservoirs.struct_reservoir import StructReservoir
from darts.tools.gen_cpg_grid import gen_cpg_grid
from iapws import SeaWater
from iapws.iapws97 import _Bound_Ph, _Region1, _Region4, _TSat_P
from xarray import DataArray, Dataset


class XarrayApi:
    def __init__(
        self,
        nx: int,
        ny: int,
        total_time: float,
        report_time: float,
        number_of_inj_wells: int,
        number_of_prd_wells: int,
        nz=None,
        last_time=False,
    ):
        """Constructor of the XarrayApi class



        :param nx: the number of blocks in x direction
        :type nx: int
        :param ny: the number of blocks in y direction
        :type ny: int
        :param total_time: the total simulation time e.g. 10x365 days
        :type total_time: int
        :param report_time: the time step reported in the time_data_report e.g. 365 days
        :type report_time: int
        :param number_of_inj_wells: the number of injection wells
        :type number_of_inj_wells: int
        :param number_of_prd_wells: the number production wells
        :type number_of_prd_wells: int
        :param nz: the number of blocks in z direction, if it is a 2D model, nz is None
        :type nz: int or None
        :return: An instance of XarrayApi
        :rtype:
        """
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.report_time = report_time
        self.total_time = total_time
        self.number_of_inj_wells = number_of_inj_wells
        self.number_of_prd_wells = number_of_prd_wells
        self.last_time = last_time

        data_vars, coords = self.create_dataset(
            int(total_time / report_time) + 1, last_time
        )
        # add more field names for dataset
        self.set_data_vars_for_wells(data_vars)

        self.xarray_dataset = Dataset(data_vars=data_vars, coords=coords)

    def create_dataset(self, number_of_time_steps, last_time=False):
        common_dims = ["Y", "X", "Z"] if self.nz is not None else ["Y", "X"]
        time_dim = "last_time" if last_time else "time"
        number_of_time_steps = 2 if last_time else number_of_time_steps

        data_vars = {
            "Perm": (
                common_dims,
                np.zeros((self.ny, self.nx, self.nz))
                if self.nz is not None
                else np.zeros((self.ny, self.nx)),
            ),
            "Poro": (
                common_dims,
                np.zeros((self.ny, self.nx, self.nz))
                if self.nz is not None
                else np.zeros((self.ny, self.nx)),
            ),
            "Pressure": (
                [*common_dims, time_dim],
                np.zeros((self.ny, self.nx, self.nz, number_of_time_steps))
                if self.nz is not None
                else np.zeros((self.ny, self.nx, number_of_time_steps)),
            ),
            "Temperature": (
                [*common_dims, time_dim],
                np.zeros((self.ny, self.nx, self.nz, number_of_time_steps))
                if self.nz is not None
                else np.zeros((self.ny, self.nx, number_of_time_steps)),
            ),
        }

        coords = {
            "time": range(number_of_time_steps),
            "last_time": range(2),
            "X": range(self.nx),
            "Y": range(self.ny),
        }

        if self.nz is not None:
            coords["Z"] = range(self.nz)

        return data_vars, coords

    def set_data_vars_for_wells(self, data_vars):
        """Set the more field names relevant to the wells in dataset

        :param data_vars: the dictionary which contains all fields in dataset
        :type data_vars: dict
        :return:
        :rtype:
        """
        # According to the number of wells to adjust the output
        # for the well information from time data report
        # injection wells
        for inj_well in range(self.number_of_inj_wells):
            data_vars[f"Injection Well {inj_well + 1} Temperature"] = (
                ["time"],
                np.zeros(int(self.total_time / self.report_time) + 1),
            )
            data_vars[f"Injection Well {inj_well + 1} Pressure"] = (
                ["time"],
                np.zeros(int(self.total_time / self.report_time) + 1),
            )
            data_vars[f"Injection Well {inj_well + 1} Energy"] = (
                ["time"],
                np.zeros(int(self.total_time / self.report_time) + 1),
            )
        # production wells
        for prd_well in range(self.number_of_prd_wells):
            data_vars[f"Production Well {prd_well + 1} Temperature"] = (
                ["time"],
                np.zeros(int(self.total_time / self.report_time) + 1),
            )
            data_vars[f"Production Well {prd_well + 1} Pressure"] = (
                ["time"],
                np.zeros(int(self.total_time / self.report_time) + 1),
            )
            data_vars[f"Production Well {prd_well + 1} Energy"] = (
                ["time"],
                np.zeros(int(self.total_time / self.report_time) + 1),
            )

    def set_xarray_static(self, property_name, property_values):
        """Set the values for static properties, e.g. permeability and porosity,
        currently can only be 'Perm' and 'Poro'

        :param property_name: the name of the static properties
        :type property_name: str
        :param property_values: numpy array of the property values
        :type property_values: np.ndarray
        :return:
        :rtype:
        """
        if property_name not in ["Perm", "Poro"]:
            raise ValueError(f"{property_name} is not in the xarray dataset...")

        self.xarray_dataset[property_name][:] = (
            DataArray(
                property_values,
                dims=("Y", "X"),
                coords={
                    "X": range(self.nx),
                    "Y": range(self.ny),
                },
            ).astype(np.float32)
            if self.nz is None
            else DataArray(
                property_values,
                dims=("Y", "X", "Z"),
                coords={"X": range(self.nx), "Y": range(self.ny), "Z": range(self.nz)},
            ).astype(np.float32)
        )

    def set_dynamic_xarray(self, wells, time_data_report):
        """Extract the information from time data report

        :param wells: the well information from DARTS reservoir class
        :type wells: list
        :param time_data_report: the time data report from the DARTS simulation
        :type time_data_report: pd.dataframe
        :return:
        :rtype:
        """
        if len(wells) != self.number_of_prd_wells + self.number_of_inj_wells:
            raise ValueError(
                "The number of production and injection wells is not consistent "
                "with the wells in DARTS model..."
            )
        for _, w in enumerate(wells):
            if w.name.lower().startswith("i"):
                self.xarray_dataset[f"Injection Well {w.name[-1]} Temperature"][
                    1:
                ] = DataArray(
                    np.array(time_data_report[f"{w.name} : temperature (K)"]),
                    dims="time",
                    coords={
                        "time": range(1, int(self.total_time / self.report_time) + 1)
                    },
                ).astype(
                    np.float32
                )
                self.xarray_dataset[f"Injection Well {w.name[-1]} Pressure"][
                    1:
                ] = DataArray(
                    np.array(time_data_report[f"{w.name} : BHP (bar)"]),
                    dims="time",
                    coords={
                        "time": range(1, int(self.total_time / self.report_time) + 1)
                    },
                ).astype(
                    np.float32
                )
                self.xarray_dataset[f"Injection Well {w.name[-1]} Energy"][
                    1:
                ] = DataArray(
                    np.array(time_data_report[f"{w.name} : energy (kJ/day)"]),
                    dims="time",
                    coords={
                        "time": range(1, int(self.total_time / self.report_time) + 1)
                    },
                ).astype(
                    np.float32
                )
            elif w.name.lower().startswith("p"):
                self.xarray_dataset[f"Production Well {w.name[-1]} Temperature"][
                    1:
                ] = DataArray(
                    np.array(time_data_report[f"{w.name} : temperature (K)"]),
                    dims="time",
                    coords={
                        "time": range(1, int(self.total_time / self.report_time) + 1)
                    },
                ).astype(
                    np.float32
                )
                self.xarray_dataset[f"Production Well {w.name[-1]} Pressure"][
                    1:
                ] = DataArray(
                    np.array(time_data_report[f"{w.name} : BHP (bar)"]),
                    dims="time",
                    coords={
                        "time": range(1, int(self.total_time / self.report_time) + 1)
                    },
                ).astype(
                    np.float32
                )
                self.xarray_dataset[f"Production Well {w.name[-1]} Energy"][
                    1:
                ] = DataArray(
                    np.array(time_data_report[f"{w.name} : energy (kJ/day)"]),
                    dims="time",
                    coords={
                        "time": range(1, int(self.total_time / self.report_time) + 1)
                    },
                ).astype(
                    np.float32
                )
            else:
                raise ValueError(
                    f"{w.name} doesn't start with 'i' or 'p', "
                    f"please check the well names..."
                )

    def set_dynamic_grid_xarray(
        self, time_index, dynamic_property_name, dynamic_property_values
    ):
        """Set the values relevant to each grid in different time steps

        :param time_index: the index of the report time
        :type time_index: int
        :param dynamic_property_name: the name of the property changing in time
        :type dynamic_property_name: str
        :param dynamic_property_values: the values of the property in one time step
        :type dynamic_property_values: np.ndarray
        :return:
        :rtype:
        """
        if dynamic_property_name not in ["Pressure", "Temperature"]:
            raise ValueError(f"{dynamic_property_name} is not in the xarray dataset...")

        if self.nz is None:
            coords = {"X": range(self.nx), "Y": range(self.ny)}
            dims = ("Y", "X")
        else:
            coords = {"X": range(self.nx), "Y": range(self.ny), "Z": range(self.nz)}
            dims = ("Y", "X", "Z")

        data_array = DataArray(
            dynamic_property_values,
            dims=dims,
            coords=coords,
        ).astype(np.float32)

        if self.nz is not None:
            # output 3D dynamic grid information
            self.xarray_dataset[dynamic_property_name][:, :, :, time_index] = data_array
        else:
            # output 2D dynamic grid information
            self.xarray_dataset[dynamic_property_name][:, :, time_index] = data_array

    def write_xarray(self, file_name, output_dir):
        """Write the xarray to .nc file

        :param file_name: the name of the nc file
        :type file_name: str
        :param output_dir: the directory where the nc file is saved
        :type output_dir: str
        :return:
        :rtype:
        """

        print(f"Saving dataset to the file {file_name}.nc")
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        self.xarray_dataset.to_netcdf(path=os.path.join(output_dir, f"{file_name}.nc"))
        print(f"Finished saving {file_name}...")

        return 0

    @staticmethod
    def get_xarray(file_name):
        """

        :param file_name: the name of the nc file
        :type file_name: str
        :return: xarray dataset
        :rtype: xarray.Dataset
        """

        xarray_data = xr.open_dataset(file_name)

        return xarray_data


class Model(DartsModel):
    def __init__(
        self,
        set_nx,
        set_ny,
        set_nz,
        perms,
        poro,
        set_dx,
        set_dy,
        set_dz,
        overburden,
        n_points=32,
    ):
        # call base class constructor
        super().__init__()

        self.timer.node["initialization"].start()
        self.report_time = 365
        self.total_time = 30 * 365
        self.nx = set_nx
        self.ny = set_ny
        self.nz = set_nz
        self.dx = set_dx
        self.dy = set_dy
        self.dz = set_dz
        self.perm = perms
        self.poro = poro
        self.overburden = overburden
        # self.set_reservoir()

        self.set_cpg_reservoir()
        self.set_physics(n_points=n_points)
        self.set_sim_params(
            first_ts=1e-5,
            mult_ts=8,
            max_ts=365,
            runtime=self.total_time,
            tol_newton=1e-3,
            tol_linear=1e-4,
            it_newton=20,
            it_linear=40,
            newton_type=sim_params.newton_global_chop,
            newton_params=value_vector([1]),
        )

        self.timer.node["initialization"].stop()

        # T_init = 350.
        # state_init = value_vector([200., 0.])
        # enth_init = self.physics.property_containers[0].enthalpy_ev['total'](T_init).evaluate(state_init)
        # self.initial_values = {self.physics.vars[0]: state_init[0],
        #                        self.physics.vars[1]: enth_init
        #                        }

    def set_cpg_reservoir(self):
        dz_list = [15, 30, 60, 120]
        underburden = self.overburden
        self.nz += self.overburden + underburden
        overburden_prop = np.ones(self.nx * self.ny * self.overburden) * 1e-5
        underburden_prop = np.ones(self.nx * self.ny * underburden) * 1e-5
        perm = np.concatenate([overburden_prop, self.perm, underburden_prop])
        poro = np.concatenate([overburden_prop, self.poro, underburden_prop])
        output = gen_cpg_grid(
            self.nx,
            self.ny,
            self.nz,
            self.dx,
            self.dy,
            self.dz,
            permx=perm,
            permy=perm,
            permz=perm,
            poro=poro,
            start_z=1855,
            burden_dz=dz_list,
        )
        coord = output["COORD"]
        zcorn = output["ZCORN"]
        arrays = {
            "SPECGRID": np.array([self.nx, self.ny, self.nz]),
            "PERMX": perm,
            "PERMY": perm,
            "PERMZ": 0.1 * perm,
            "PORO": poro,
            "COORD": coord,
            "ZCORN": zcorn,
        }
        arrays["ACTNUM"] = np.ones(arrays["SPECGRID"].prod(), dtype=np.int32)

        self.reservoir = CPG_Reservoir(self.timer, arrays)
        self.reservoir.discretize()
        # add larger volumes
        self.reservoir.set_boundary_volume(
            xz_minus=1e8, xz_plus=1e8, yz_minus=1e8, yz_plus=1e8
        )
        self.reservoir.apply_volume_depth()

        # get wrapper around local array (length equal to active blocks number)
        cond_mesh = np.array(self.reservoir.mesh.rock_cond, copy=False)
        hcap_mesh = np.array(self.reservoir.mesh.heat_capacity, copy=False)
        cond_mesh[np.array(self.reservoir.mesh.poro) <= 0.1] = 2.2 * 86.4
        cond_mesh[np.array(self.reservoir.mesh.poro) > 0.1] = 3 * 86.4
        hcap_mesh[np.array(self.reservoir.mesh.poro) <= 0.1] = 2300
        hcap_mesh[np.array(self.reservoir.mesh.poro) > 0.1] = 2450

    def set_initial_conditions(
        self, initial_values: dict = None, gradient: dict = None
    ):
        self.physics.set_nonuniform_initial_conditions(
            self.reservoir.mesh, pressure_grad=107, temperature_grad=30
        )

    def set_reservoir(self):
        overburden_dz = np.zeros(self.nx * self.ny * self.overburden)
        underburden_dz = np.zeros(self.nx * self.ny * self.overburden)
        dz_list = [15, 30, 60, 120]
        # dz_list = []
        if self.overburden == 0:
            set_dz_new = self.dz
        else:
            for k in range(self.overburden):
                start = k * self.nx * self.ny
                end = (k + 1) * self.nx * self.ny
                if k == 0:
                    overburden_dz[start:end] = dz_list[3]
                    underburden_dz[start:end] = dz_list[0]
                elif k == 1:
                    overburden_dz[start:end] = dz_list[2]
                    underburden_dz[start:end] = dz_list[1]
                elif k == 2:
                    overburden_dz[start:end] = dz_list[1]
                    underburden_dz[start:end] = dz_list[2]
                else:
                    overburden_dz[start:end] = dz_list[0]
                    underburden_dz[start:end] = dz_list[3]

            # add the overburden layers' dz
            set_dz_new = np.concatenate(
                [
                    overburden_dz,
                    np.ones(self.nx * self.ny * self.nz) * self.dz,
                    underburden_dz,
                ]
            )

        underburden = self.overburden
        self.nz += self.overburden + underburden
        overburden_prop = np.ones(self.nx * self.ny * self.overburden) * 0.001
        underburden_prop = np.ones(self.nx * self.ny * underburden) * 0.001
        perm = np.concatenate([overburden_prop, self.perm, underburden_prop])
        poro = np.concatenate([overburden_prop, self.poro, underburden_prop])

        # gen_cpg_grid(self.nx, self.ny, self.nz-(self.overburden + underburden), self.dx, self.dy, self.dz, permx=perm, permy=perm, permz=perm, poro=poro,burden_dz=dz_list, gridname='str_reservoir.grdecl')

        self.xrdata = XarrayApi(
            nx=self.nx,
            ny=self.ny,
            nz=self.nz,
            total_time=self.total_time,
            report_time=self.report_time,
            number_of_inj_wells=1,
            number_of_prd_wells=1,
            last_time=False,
        )
        self.xrdata.set_xarray_static(
            "Perm", np.rot90(perm.reshape((self.nx, self.ny, self.nz), order="F"))
        )
        self.xrdata.set_xarray_static(
            "Poro", np.rot90(poro.reshape((self.nx, self.ny, self.nz), order="F"))
        )
        # rock heat capacity and rock thermal conduction
        hcap = np.ones(self.nx * self.ny * self.nz)
        rcond = np.ones(self.nx * self.ny * self.nz)
        hcap[poro <= 1e-1] = 2300  # volumetric heat capacity: kJ/m3/K
        hcap[poro > 1e-1] = 2450

        rcond[poro <= 1e-1] = 2.2 * 86.4  # kJ/m/day/K
        rcond[poro > 1e-1] = 3 * 86.4

        # depth = np.empty(self.nx * self.ny * self.nz)
        # top = 1630
        # unique_thickness = sorted(np.unique(set_dz_new))
        # if len(unique_thickness) > 1:
        #     for k in range(self.nz):
        #         if k < 4:
        #             depth[k * self.nx * self.ny:(k + 1) * self.nx * self.ny] = top + sum(
        #                 unique_thickness[-i] for i in range(1, k + 2, 1)) / 2
        #         elif 4 <= k <= 19:
        #             depth[k * self.nx * self.ny:(k + 1) * self.nx * self.ny] = top + 225 + (2 * (k - 3) - 1) * \
        #                                                                        unique_thickness[0] * 0.5
        #         elif 19 < k <= 22:
        #             depth[k * self.nx * self.ny:(k + 1) * self.nx * self.ny] = top + 225 + 100 + sum(
        #                 unique_thickness[-i] for i in range(1, 5, 1)) / 2
        # else:
        #     for k in range(self.nz):
        #         depth[k * self.nx * self.ny:(k + 1) * self.nx * self.ny] = top + (2 * k + 1) * unique_thickness[0] * 0.5

        # add more layers above or below the reservoir
        self.reservoir = StructReservoir(
            self.timer,
            nx=self.nx,
            ny=self.ny,
            nz=self.nz,
            dx=self.dx,
            dy=self.dy,
            dz=set_dz_new,
            permx=perm,
            permy=perm,
            permz=0.1 * perm,
            poro=poro,
            depth=None,
        )

        # add larger volumes
        # self.reservoir.boundary_volumes['xy_minus'] = 1e18
        # self.reservoir.boundary_volumes['xy_plus'] = 1e18
        self.reservoir.boundary_volumes["yz_minus"] = 1e8
        self.reservoir.boundary_volumes["yz_plus"] = 1e8
        self.reservoir.boundary_volumes["xz_minus"] = 1e8
        self.reservoir.boundary_volumes["xz_plus"] = 1e8

        return

    def set_wells(self):
        # add well's locations
        injection_well_x = int(2400 / self.dx)
        production_well_x = injection_well_x + int(1300 / self.dx)

        # add perforations to the payzone
        start_index = self.overburden + 1
        end_index = self.nz - self.overburden + 1
        # add well
        self.reservoir.add_well("INJ1")
        for k in range(start_index, end_index):
            self.reservoir.add_perforation(
                "INJ1",
                cell_index=(injection_well_x, int(self.ny / 2), k),
                well_radius=0.16,
                verbose=True,
            )

        self.reservoir.add_well("PRD1")
        for k in range(start_index, end_index):
            self.reservoir.add_perforation(
                "PRD1",
                cell_index=(production_well_x, int(self.ny / 2), k),
                well_radius=0.16,
                verbose=True,
            )

        return

    def set_physics(self, n_points):
        # create pre-defined physics for geothermal
        property_container = NewPropertyContainer()
        # components = ["H2O", "NaCl"]
        # property_container.density_ev = dict([('water', Spivey2004(components=components, molarity=250/58.44,ions=['Na','Cl'])),
        #                                       ('steam', iapws_steam_density_evaluator())])
        # property_container.viscosity_ev = dict([('water', MaoDuan2009(components=components, molarity=250/58.44)),
        #                                         ('steam', iapws_steam_viscosity_evaluator())])
        # Define property evaluators with constant density and viscosity
        # property_container.density_ev = dict([('water', DensityBasic(compr=1e-5, dens0=54.53)),
        #                                      ('steam', DensityBasic(compr=1e-5, dens0=0)),
        #                                     ])
        # property_container.viscosity_ev = dict([('water', ConstFunc(0.38)),
        #                                         ('steam', ConstFunc(0.1))])
        self.physics = Geothermal(
            self.timer,
            n_points=n_points,
            min_p=100,
            max_p=500,
            min_e=1000,
            max_e=25000,
            cache=False,
            mass_rate=True,
        )
        self.physics.add_property_region(property_container)

        return

    # T=300K, P=200bars, the enthalpy is 2358 [kJ/kmol]
    def set_well_controls(self):
        self.physics.define_well_controls()
        for i, w in enumerate(self.reservoir.wells):
            if w.name.lower().startswith("i"):
                # w.control = self.physics.new_rate_water_inj(7500, 300)
                w.control = self.physics.new_mass_rate_water_inj(417000, 2358)
                # w.constraint = self.physics.new_bhp_water_inj(200, self.inj_temperature)
            else:
                # w.control = self.physics.new_rate_water_prod(7500)
                w.control = self.physics.new_mass_rate_water_prod(417000)


class NewPropertyContainer(PropertyContainer):
    def evaluate(self, state):
        self.temperature = self.temperature_ev.evaluate(state)

        for j, phase in enumerate(["water", "steam"]):
            self.saturation[j] = self.saturation_ev[phase].evaluate(state)
            self.conduction[j] = self.conduction_ev[phase].evaluate(state)
            # self.density[j] = self.density_ev[phase].evaluate(state[0], self.temperature)
            self.density[j] = self.density_ev[phase].evaluate(state)
            self.viscosity[j] = self.viscosity_ev[phase].evaluate(state)
            # self.viscosity[j] = self.viscosity_ev[phase].evaluate(state[0], self.temperature)
            self.relperm[j] = self.relperm_ev[phase].evaluate(state)
            self.enthalpy[j] = self.enthalpy_ev[phase].evaluate(state)
        # self.viscosity[0] = self.viscosity_ev['water'].evaluate(state[0], self.temperature)
        # self.viscosity[1] = self.viscosity_ev['steam'].evaluate(state)
        #
        # self.density[0] = self.density_ev['water'].evaluate(state[0], self.temperature)/18.015
        # self.density[1] = self.density_ev['steam'].evaluate(state)
        return


class seawater_enthalpy_evaluator(property_evaluator_iface):
    def __init__(self, brine_mass_fraction):
        super().__init__()
        self.brine_mass_fraction = brine_mass_fraction

    def evaluate(self, pressure, temperature):
        seawater = SeaWater(T=temperature, P=pressure / 10, S=self.brine_mass_fraction)
        seawater_enth = seawater.h

        return seawater_enth * 18.015


class seawater_density_evaluator(property_evaluator_iface):
    def __init__(self, brine_mass_fraction):
        super().__init__()
        self.brine_mass_fraction = brine_mass_fraction

    def evaluate(self, pressure, temperature):
        seawater = SeaWater(T=temperature, P=pressure / 10, S=self.brine_mass_fraction)
        seawater_enth = seawater.rho

        return seawater_enth / 18.015


class brine_density_evaluator(property_evaluator_iface):
    def __init__(self, brine_mass_fraction):
        super().__init__()
        self.brine_mass_fraction = brine_mass_fraction

    def evaluate(self, state):
        P, h = state[0] * 0.1, state[1] / 18.015
        Pmin = 0.000611212677444
        if P < Pmin:
            P = Pmin
        hmin = _Region1(273.15, P)["h"]
        if h < hmin:
            h = hmin

        region = _Bound_Ph(P, h)
        if region == 1:
            temperature = temperature_region1_evaluator()
            T = temperature.evaluate(state)
            water_density = (
                iapws_water_density_evaluator().evaluate(state) * 18.015 / 1000
            )  # convert to g/cm3
        elif region == 4:
            T = _TSat_P(P)
            if T <= 623.15:
                water_density = 1 / _Region4(P, 0)["v"] / 1000  # convert to g/cm3
            else:
                raise NotImplementedError(
                    "Variables out of bound: p="
                    + str(P)
                    + "h="
                    + str(h)
                    + " region="
                    + str(region)
                )
        elif region == 2:
            water_density = 0
        else:
            raise NotImplementedError(
                "Variables out of bound: p="
                + str(P)
                + "h="
                + str(h)
                + " region="
                + str(region)
            )
        T -= 273.15
        brine_density = water_density + self.brine_mass_fraction * (
            0.668
            + 0.44 * self.brine_mass_fraction
            + 1e-6
            * (
                300 * P
                - 2400 * P * self.brine_mass_fraction
                + T
                * (
                    80
                    + 3 * T
                    - 3300 * self.brine_mass_fraction
                    - 13 * P
                    + 47 * P * self.brine_mass_fraction
                )
            )
        )

        return brine_density * 1000 / 18.015


class brine_viscosity_evaluator(property_evaluator_iface):
    def __init__(self, brine_mass_fraction):
        super().__init__()
        self.brine_mass_fraction = brine_mass_fraction

    def evaluate(self, state):
        P, h = state[0] * 0.1, state[1] / 18.015
        Pmin = 0.000611212677444
        if P < Pmin:
            P = Pmin
        hmin = _Region1(273.15, P)["h"]
        if h < hmin:
            h = hmin

        region = _Bound_Ph(P, h)
        if region == 1:
            temperature = temperature_region1_evaluator()
            T = temperature.evaluate(state)
        T -= 273.15
        brine_viscosity = (
            0.1
            + 0.333 * self.brine_mass_fraction
            + (1.65 + 91.9 * self.brine_mass_fraction**3)
            * np.exp(
                -(
                    0.42 * ((np.power(self.brine_mass_fraction, 0.8) - 0.17) ** 2)
                    + 0.045
                )
                * np.power(T, 0.8)
            )
        )

        return brine_viscosity
